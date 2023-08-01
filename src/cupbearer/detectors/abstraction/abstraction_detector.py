import functools
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from cupbearer.detectors.abstraction.abstraction import (
    Abstraction,
    abstraction_collate,
    get_default_abstraction,
)
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.models.computations import Model
from cupbearer.utils import trainer, utils
from cupbearer.utils.optimizers import OptimizerConfig


def kl_loss_fn(predicted_logits, logits):
    # KL divergence between actual and predicted output:
    # TODO: Should probably just use something like distrax.
    probs = jax.nn.softmax(logits, axis=-1)
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    predicted_logprobs = jax.nn.log_softmax(predicted_logits, axis=-1)
    output_losses = (probs * (logprobs - predicted_logprobs)).sum(axis=-1)
    return output_losses


def single_class_loss_fn(predicted_logits, logits, target=0):
    # This loss function only cares about correctly predicting whether the output
    # is a specific digit (like zero) or not.
    # It combines the probabilities for all non-zero classes
    # into a single "non-zero" class. Assumes that output_dim of the abstraction is 2.
    assert predicted_logits.ndim == logits.ndim == 2
    assert logits.shape[-1] == 10, logits.shape
    assert predicted_logits.shape[-1] == 2, predicted_logits.shape
    assert predicted_logits.shape[0] == logits.shape[0]

    all_probs = jax.nn.softmax(logits, axis=-1)
    target_probs = all_probs[:, target]
    non_target_probs = all_probs[:, :target].sum(axis=-1) + all_probs[
        :, target + 1 :
    ].sum(-1)
    probs = jnp.stack([non_target_probs, target_probs], axis=-1)
    logprobs = jnp.log(probs)
    predicted_logprobs = jax.nn.log_softmax(predicted_logits, axis=-1)
    output_losses = (probs * (logprobs - predicted_logprobs)).sum(axis=-1)
    return output_losses


OUTPUT_LOSS_FNS: dict[str, Callable] = {
    "kl": kl_loss_fn,
    "single_class": single_class_loss_fn,
}


def compute_losses(
    params, state, batch, output_loss_fn: Callable, return_batch=False, layerwise=False
):
    logits, activations = batch
    abstractions, predicted_abstractions, predicted_logits = state.apply_fn(
        {"params": params}, activations
    )
    norms = jax.tree_map(
        functools.partial(jnp.linalg.norm, axis=tuple(range(1, abstractions[0].ndim))),
        abstractions,
    )
    norms = jax.tree_map(lambda x: x.mean(), norms)
    avg_norm = sum(norms) / len(norms)
    assert isinstance(abstractions, list)
    assert isinstance(predicted_abstractions, list)
    assert len(abstractions) == len(predicted_abstractions) + 1
    b, *_ = abstractions[0].shape
    assert logits.shape == (b, 10)
    # Output dimension of abstraction may be different from full computation,
    # depending on the output_loss_fn. So only check batch dimension.
    assert predicted_logits.shape[0] == b

    output_losses = output_loss_fn(predicted_logits, logits)
    output_loss = output_losses.mean()

    # Consistency loss:
    layer_losses = []
    # Skip the first abstraction, since there's no prediction for that
    for actual_abstraction, predicted_abstraction in zip(
        abstractions[1:], predicted_abstractions
    ):
        actual_abstraction = actual_abstraction.reshape(b, -1)
        predicted_abstraction = predicted_abstraction.reshape(b, -1)
        # Cosine distance can be NaN if one of the inputs is exactly zero
        # which is why we need the eps (which sets cosine distance to 1 in that case).
        # This doesn't happen in realistic scenarios, but in tests with very small
        # hidden dimensions and ReLUs, it's possible.
        losses = optax.cosine_distance(
            predicted_abstraction, actual_abstraction, epsilon=1e-6
        )
        layer_losses.append(losses)

    consistency_losses = jnp.stack(layer_losses, axis=0).mean(axis=0)
    consistency_losses /= len(predicted_abstractions)

    if layerwise:
        layer_losses = jnp.stack(layer_losses, axis=0)
        if not return_batch:
            # Take mean over batch dimension
            layer_losses = layer_losses.mean(axis=1)
            layer_losses = jnp.concatenate([layer_losses, jnp.array([output_loss])])
        else:
            layer_losses = jnp.concatenate([layer_losses, output_losses[None]])
        return layer_losses

    if return_batch:
        return output_losses + consistency_losses

    consistency_loss = consistency_losses.mean()
    loss = output_loss + consistency_loss

    return loss, (output_loss, consistency_loss, avg_norm)


class AbstractionTrainer(trainer.TrainerModule):
    def __init__(
        self,
        output_loss_fn: str = "kl",
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        # The output loss function describes how to compute the loss between the
        # actual output and the predicted one. It should take a batch of predicted
        # outputs and a batch of actual outputs and return a *batch* of losses.
        # The main point is to allow ignoring certain differences in the output
        # that we aren't aiming to predict
        self.output_loss_fn = OUTPUT_LOSS_FNS[output_loss_fn]

    def create_functions(self):
        def train_step(state, batch):
            def loss_fn(params):
                return compute_losses(
                    params, state, batch, output_loss_fn=self.output_loss_fn
                )

            (loss, (output_loss, consistency_loss, _)), grads = jax.value_and_grad(
                loss_fn, has_aux=True
            )(state.params)

            state = state.apply_gradients(grads=grads)
            metrics = {
                "loss": loss,
                "output_loss": output_loss,
                "consistency_loss": consistency_loss,
            }
            return state, metrics

        def eval_step(state, batch):
            # Note that this class expects the eval dataloader to return not just
            # logits/activations, but also the original data. That's necessary
            # to compute metrics on specific clases (zero in this case).
            logits, activations = batch
            (
                loss,
                (output_loss, consistency_loss, _),
            ) = compute_losses(  # type: ignore
                state.params,
                state,
                (logits, activations),
                output_loss_fn=self.output_loss_fn,
            )
            metrics = {
                "loss": loss,
                "output_loss": output_loss,
                "consistency_loss": consistency_loss,
            }
            return metrics

        return train_step, eval_step

    def on_training_epoch_end(self, epoch_idx, metrics):
        logger.info(self._prettify_metrics(metrics))

    def on_validation_epoch_end(self, epoch_idx: int, metrics, val_loader):
        logger.info(self._prettify_metrics(metrics))

    def _prettify_metrics(self, metrics):
        return "\n" + "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())


# abstraction_state will be stored already, no need to capture it as a kwarg.
# When loading, the class will be initialized with abstraction_state=None,
# but then it will immediately be overriden.
@functools.partial(
    utils.store_init_args, ignore={"self", "model", "params", "abstraction_state"}
)
class AbstractionDetector(AnomalyDetector):
    """Anomaly detector based on an abstraction.
    States this detector can be in:
      - abstraction_state is None, abstraction is None: detector is not usable,
        needs to be trained first.
      - abstraction_state is None, but abstraction is set: if detector is used,
        abstraction_state will be randomly initialized. In particular, the detector
        can be "finetuned" in this state, based on a random initialization.
      - both are set: detector will use the abstraction_state.
    Either abstraction or size_reduction should be set if the first state
    is to be avoided.
    """

    def __init__(
        self,
        model: Model,
        params,
        abstraction: Optional[Abstraction] = None,
        abstraction_state: Optional[trainer.InferenceState | trainer.TrainState] = None,
        size_reduction: Optional[int] = None,
        abstraction_cls: type[Abstraction] = Abstraction,
        output_loss_fn: str = "kl",
        max_batch_size: int = 4096,
        save_path: str | Path | None = None,
    ):
        if abstraction is None and size_reduction is not None:
            abstraction = get_default_abstraction(
                model,
                size_reduction,
                output_dim=2 if output_loss_fn == "single_class" else None,
                abstraction_cls=abstraction_cls,
            )
        self.abstraction = abstraction
        self.abstraction_state = abstraction_state
        self.output_loss_fn = output_loss_fn
        super().__init__(
            model, params, max_batch_size=max_batch_size, save_path=save_path
        )

    def train(
        self,
        dataset,
        optimizer: OptimizerConfig,
        batch_size: int = 128,
        num_epochs: int = 10,
        validation_datasets: Optional[dict[str, Dataset]] = None,
        max_steps: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ):
        assert self.abstraction is not None
        # First sample, only input without label.
        # Also need to add a batch dimension
        example_input = dataset[0][0][None]
        _, example_activations = self._model(example_input)
        trainer = AbstractionTrainer(
            model=self.abstraction,
            output_loss_fn=self.output_loss_fn,
            example_input=example_activations,
            log_dir=self.save_path,
            optimizer=optimizer.build(),
            **kwargs,
        )
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            # TODO: I think this should plausibly be handled in AbstractionTrainer
            # now after the refactor.
            # Or maybe AbstractionTrainer and AbstractionDetector should just be merged.
            collate_fn=abstraction_collate(self.model, self.params),
        )
        val_loaders = {}
        if validation_datasets is not None:
            for key, ds in validation_datasets.items():
                val_loaders[key] = DataLoader(
                    dataset=ds,
                    batch_size=self.max_batch_size,
                    collate_fn=abstraction_collate(self.model, self.params),
                )
        trainer.train_model(
            train_loader=train_loader,
            val_loaders=val_loaders,
            test_loaders=None,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )
        trainer.close_loggers()

        self.abstraction_state = trainer.state

    def layerwise_scores(self, batch):
        assert self.abstraction is not None
        batch = self._model(batch)
        if self.abstraction_state is None:
            logger.info("Randomly initializing abstraction.")
            # TODO: should derive this from some other key to avoid
            # hard-coding seed.
            model_rng = jax.random.PRNGKey(0)
            model_rng, init_rng = jax.random.split(model_rng)
            output, activations = batch
            variables = self.abstraction.init(init_rng, activations)
            self.abstraction_state = trainer.InferenceState(
                self.abstraction.apply,
                params=variables["params"],
                batch_stats=variables.get("batch_stats", None),
                rng=model_rng,
            )
        return compute_losses(
            params=self.abstraction_state.params,
            state=self.abstraction_state,
            batch=batch,
            output_loss_fn=OUTPUT_LOSS_FNS[self.output_loss_fn],
            return_batch=True,
            layerwise=True,
        )

    def _get_drawable(self, layer_scores, inputs):
        assert self.abstraction is not None
        return self.abstraction.get_drawable(
            full_model=self.model, layer_scores=layer_scores, inputs=inputs
        )

    def _get_trained_variables(self):
        return self._state_to_dict(self.abstraction_state)

    def _set_trained_variables(self, variables):
        assert self.abstraction is not None
        assert variables
        # TODO: in general we might have to create a PRNG here as well
        self.abstraction_state = trainer.InferenceState(
            self.abstraction.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
        )

    def _state_to_dict(self, state: trainer.TrainState | trainer.InferenceState | None):
        if state is None:
            return {}
        return {
            "params": state.params,
            "batch_stats": state.batch_stats,
        }
