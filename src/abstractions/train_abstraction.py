from pathlib import Path
import sys
import flax.linen as nn
from typing import Callable, Optional
from abstractions.anomaly_detector import AnomalyDetector
from abstractions.computations import Computation, Step, get_abstraction_maps
from torch.utils.data import DataLoader, Dataset
import hydra
import jax
import jax.numpy as jnp
from jax.config import config as jax_config
from loguru import logger
import sklearn.metrics
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from abstractions import abstraction, data, utils, trainer
from abstractions.abstraction import Abstraction, Model
from abstractions.logger import WandbLogger, DummyLogger


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
    # is a specific digit (like zero) or not. It combines the probabilities for all non-zero classes into
    # a single "non-zero" class. Assumes that output_dim of the abstraction is 2.
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


def compute_losses(
    params, state, batch, output_loss_fn, mask=None, return_batch=False, layerwise=False
):
    logits, activations = batch
    abstractions, predicted_abstractions, predicted_logits = state.apply_fn(
        {"params": params}, activations
    )
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
    for abstraction, predicted_abstraction in zip(
        abstractions[1:], predicted_abstractions
    ):
        # Take mean over hidden dimension(s):
        layer_losses.append(
            jnp.sqrt(
                ((abstraction - predicted_abstraction) ** 2).mean(
                    axis=tuple(range(1, abstraction.ndim))
                )
            )
        )

    consistency_losses = jnp.stack(layer_losses, axis=0).mean(axis=0)
    consistency_losses /= len(predicted_abstractions)

    if layerwise:
        assert mask is None
        layer_losses = jnp.stack(layer_losses, axis=0)
        if not return_batch:
            # Take mean over batch dimension
            layer_losses = layer_losses.mean(axis=1)
            layer_losses = jnp.concatenate([layer_losses, jnp.array([output_loss])])
        else:
            layer_losses = jnp.concatenate([layer_losses, output_losses[None]])
        return layer_losses

    if return_batch:
        assert mask is None
        return output_losses + consistency_losses

    consistency_loss = consistency_losses.mean()
    loss = output_loss + consistency_loss

    if mask is not None:
        assert mask.shape == (b,)
        num_samples = mask.sum()
        output_losses *= mask
        # Can't take mean here because we don't want to count masked samples
        masked_output_loss = output_losses.sum() / num_samples
        consistency_losses *= mask
        masked_consistency_loss = consistency_losses.sum() / num_samples
        masked_loss = masked_output_loss + masked_consistency_loss

        return (
            loss,
            (output_loss, consistency_loss),
            masked_loss,
            (
                masked_output_loss,
                masked_consistency_loss,
            ),
        )

    return loss, (output_loss, consistency_loss)


class AbstractionTrainer(trainer.TrainerModule):
    def __init__(
        self,
        output_loss_fn: Callable = kl_loss_fn,
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
        self.output_loss_fn = output_loss_fn

    def create_functions(self):
        def train_step(state, batch):
            loss_fn = lambda params: compute_losses(
                params, state, batch, output_loss_fn=self.output_loss_fn
            )
            (loss, (output_loss, consistency_loss)), grads = jax.value_and_grad(
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
            logits, activations, (images, labels, infos) = batch
            zeros = infos["original_target"] == 0
            (
                loss,
                (output_loss, consistency_loss),
                zeros_loss,
                (zeros_output_loss, zeros_consistency_loss),
            ) = compute_losses(  # type: ignore
                state.params,
                state,
                (logits, activations),
                output_loss_fn=self.output_loss_fn,
                mask=zeros,
            )
            metrics = {
                "loss": loss,
                "output_loss": output_loss,
                "consistency_loss": consistency_loss,
                "zeros_loss": zeros_loss,
                "zeros_output_loss": zeros_output_loss,
                "zeros_consistency_loss": zeros_consistency_loss,
            }
            return metrics

        return train_step, eval_step

    def on_training_epoch_end(self, epoch_idx, metrics):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def on_validation_epoch_end(self, epoch_idx: int, metrics, val_loader):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def _prettify_metrics(self, metrics):
        return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())


class AbstractionDetector(AnomalyDetector):
    def __init__(
        self,
        model: Model,
        params,
        abstraction: Optional[Abstraction] = None,
        abstraction_state: Optional[trainer.InferenceState | trainer.TrainState] = None,
        output_loss_fn: Callable = kl_loss_fn,
        max_batch_size: int = 4096,
    ):
        self.abstraction = abstraction
        self.abstraction_state = abstraction_state
        self.output_loss_fn = output_loss_fn
        super().__init__(model, params, max_batch_size=max_batch_size)

    def train(
        self,
        dataset,
        trainer: AbstractionTrainer,
        batch_size: int = 128,
        num_epochs: int = 10,
        validation_datasets: Optional[dict[str, Dataset]] = None,
    ):
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            # TODO: I think this should plausibly be handled in AbstractionTrainer
            # now after the refactor. Or maybe AbstractionTrainer and AbstractionDetector
            # should just be merged.
            collate_fn=abstraction.abstraction_collate(self.model, self.params),
        )
        val_loaders = {}
        if validation_datasets is not None:
            for key, ds in validation_datasets.items():
                val_loaders[key] = DataLoader(
                    dataset=ds,
                    batch_size=self.max_batch_size,
                    collate_fn=abstraction.abstraction_collate(
                        self.model, self.params, return_original_batch=True
                    ),
                )
        trainer.train_model(
            train_loader=train_loader,
            val_loaders=val_loaders,
            test_loaders=None,
            num_epochs=num_epochs,
        )

        self.abstraction = trainer.model
        self.abstraction_state = trainer.state
        self.output_loss_fn = trainer.output_loss_fn

    def layerwise_scores(self, batch):
        assert self.abstraction_state is not None
        return compute_losses(
            params=self.abstraction_state.params,
            state=self.abstraction_state,
            batch=self._model(batch),
            output_loss_fn=self.output_loss_fn,
            return_batch=True,
            layerwise=True,
        )

    def _get_drawable(self, layer_scores, inputs):
        assert self.abstraction is not None
        return self.abstraction.get_drawable(
            full_model=self.model, layer_scores=layer_scores, inputs=inputs
        )

    def _get_trained_variables(self):
        assert self.abstraction_state is not None
        return {
            "params": self.abstraction_state.params,
            "batch_stats": self.abstraction_state.batch_stats,
        }

    def _set_trained_variables(self, variables):
        assert self.abstraction is not None
        # TODO: in general we might have to create a PRNG here as well
        self.abstraction_state = trainer.InferenceState(
            self.abstraction.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
        )


@hydra.main(version_base=None, config_path="conf", config_name="abstraction")
def train_and_evaluate(cfg: DictConfig):
    """Execute model training and evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    if cfg.debug:
        jax_config.update("jax_debug_nans", True)
        jax_config.update("jax_disable_jit", True)
        cfg.wandb = False

    if cfg.wandb:
        metrics_logger = WandbLogger(
            project_name="abstractions",
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )
    else:
        metrics_logger = DummyLogger()

    # Load the full model we want to abstract
    base_run = Path(cfg.base_run)
    base_cfg = OmegaConf.load(
        to_absolute_path(str(base_run / ".hydra" / "config.yaml"))
    )

    full_computation = hydra.utils.call(base_cfg.model)
    full_model = abstraction.Model(computation=full_computation)
    full_params = utils.load(to_absolute_path(str(base_run / "model.pytree")))["params"]

    train_dataset = data.get_dataset(base_cfg.train_data)

    # First sample, only input without label and info. Also need to add a batch dimension
    example_input = train_dataset[0][0][None]
    _, example_activations = full_model.apply(
        {"params": full_params}, example_input, return_activations=True
    )

    if cfg.single_class:
        cfg.model.output_dim = 2

    computation = hydra.utils.call(cfg.model)
    # TODO: Might want to make this configurable somehow, but it's a reasonable
    # default for now
    maps = get_abstraction_maps(cfg.model)
    model = abstraction.Abstraction(computation=computation, abstraction_maps=maps)

    trainer = AbstractionTrainer(
        model=model,
        output_loss_fn=single_class_loss_fn if cfg.single_class else kl_loss_fn,
        optimizer=hydra.utils.instantiate(cfg.optim),
        example_input=example_activations,
        # Hydra sets the cwd to the right log dir automatically
        log_dir=".",
        check_val_every_n_epoch=1,
        loggers=[metrics_logger],
        enable_progress_bar=False,
    )
    detector = AbstractionDetector(
        model=full_model,
        params=full_params,
        max_batch_size=cfg.max_batch_size,
    )

    val_datasets = {k: data.get_dataset(v) for k, v in cfg.val_data.items()}

    detector.train(
        train_dataset,
        trainer,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        validation_datasets=val_datasets,
    )

    detector.save("detector")

    trainer.close_loggers()


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, color="<green>", icon="📈")
    logger.add(
        sys.stderr, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stderr, filter=lambda record: record["level"].name != "METRICS")
    train_and_evaluate()
