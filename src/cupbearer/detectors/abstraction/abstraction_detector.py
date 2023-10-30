from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from cupbearer.detectors.abstraction.abstraction import (
    Abstraction,
    abstraction_collate,
)
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.utils import trainer
from cupbearer.utils.optimizers import OptimizerConfig


def norm(x: torch.Tensor):
    """Compute the L2 norm along all but the first axis."""
    return torch.norm(x, p=2, dim=tuple(range(1, x.ndim)))


def compute_losses(
    abstraction_model: Abstraction,
    batch: tuple[torch.Tensor, dict[str, torch.Tensor]],
    return_batch=False,
    layerwise=False,
):
    _, activations = batch
    abstractions, predicted_abstractions = abstraction_model(activations)

    # Consistency loss:
    layer_losses = {}
    for k in abstractions:
        predicted_abstraction = predicted_abstractions[k]
        if predicted_abstraction is None:
            # We didn't make a prediction for this one
            continue
        actual_abstraction = abstractions[k]
        batch_size = actual_abstraction.shape[0]
        actual_abstraction = actual_abstraction.view(batch_size, -1)
        predicted_abstraction = predicted_abstraction.view(batch_size, -1)
        # Cosine distance can be NaN if one of the inputs is exactly zero
        # which is why we need the eps (which sets cosine distance to 1 in that case).
        # This doesn't happen in realistic scenarios, but in tests with very small
        # hidden dimensions and ReLUs, it's possible.
        predicted_abstraction = F.normalize(predicted_abstraction, dim=1, eps=1e-6)
        actual_abstraction = F.normalize(actual_abstraction, dim=1, eps=1e-6)
        losses = 1 - (predicted_abstraction * actual_abstraction).sum(dim=1)
        layer_losses[k] = losses

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
        return consistency_losses

    loss = consistency_losses.mean()

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


class AbstractionDetector(AnomalyDetector):
    """Anomaly detector based on an abstraction."""

    def __init__(
        self,
        model: Model,
        abstraction: Abstraction,
        abstraction_state: Optional[trainer.InferenceState | trainer.TrainState] = None,
        output_loss_fn: str = "kl",
        max_batch_size: int = 4096,
        save_path: str | Path | None = None,
    ):
        self.abstraction = abstraction
        self.abstraction_state = abstraction_state
        self.output_loss_fn = output_loss_fn
        super().__init__(model, max_batch_size=max_batch_size, save_path=save_path)

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
        # First sample, only input without label.
        # Also need to add a batch dimension
        example_input = dataset[0][0][None]
        _, example_activations = self._model(example_input)
        self.rng, trainer_rng = jax.random.split(self.rng)
        trainer = AbstractionTrainer(
            model=self.abstraction,
            output_loss_fn=self.output_loss_fn,
            example_input=example_activations,
            log_dir=self.save_path,
            optimizer=optimizer.build(),
            rng=trainer_rng,
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
        batch = self._model(batch)
        self._ensure_abstraction_state(batch)

        return compute_losses(
            params=self.abstraction_state.params,  # type: ignore
            state=self.abstraction_state,
            batch=batch,
            output_loss_fn=OUTPUT_LOSS_FNS[self.output_loss_fn],
            return_batch=True,
            layerwise=True,
        )

    def _ensure_abstraction_state(self, batch):
        if self.abstraction_state is None:
            logger.info("Randomly initializing abstraction.")
            self.rng, model_rng, init_rng = jax.random.split(self.rng, 3)
            output, activations = batch
            variables = self.abstraction.init(init_rng, activations)
            self.abstraction_state = trainer.InferenceState(
                self.abstraction.apply,
                params=variables["params"],
                batch_stats=variables.get("batch_stats", None),
                rng=model_rng,
            )

    def _get_drawable(self, layer_scores, inputs):
        assert self.abstraction is not None
        return self.abstraction.get_drawable(
            full_model=self.model, layer_scores=layer_scores, inputs=inputs
        )

    def _get_trained_variables(self, saving: bool = False):
        state = self.abstraction_state
        if state is None:
            return {}
        result = {
            "params": state.params,
            "batch_stats": state.batch_stats,
        }

        # We currently never save the optimizer to disk, mainly because the serizalition
        # implementation in utils.save() wouldn't work for tx.
        # TODO: now that there's pickle support, may want to revisit that.
        if isinstance(state, trainer.TrainState) and not saving:
            result["step"] = state.step
            result["tx"] = state.tx
            result["opt_state"] = state.opt_state

        return result

    def _set_trained_variables(self, variables):
        self.rng, model_rng = jax.random.split(self.rng)
        if "opt_state" in variables:
            self.abstraction_state = trainer.TrainState(
                apply_fn=self.abstraction.apply,
                step=variables["step"],
                params=FrozenDict(variables["params"]),
                batch_stats=variables["batch_stats"],
                tx=variables["tx"],
                opt_state=variables["opt_state"],
                rng=model_rng,
            )
        else:
            self.abstraction_state = trainer.InferenceState(
                apply_fn=self.abstraction.apply,
                params=FrozenDict(variables["params"]),
                batch_stats=variables["batch_stats"],
                rng=model_rng,
            )
