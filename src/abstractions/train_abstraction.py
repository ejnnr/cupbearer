import copy
import sys
from pathlib import Path
from typing import Callable, Optional

import hydra
import jax
import jax.numpy as jnp
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset

from abstractions import abstraction, data, trainer, utils
from abstractions.abstraction import Abstraction, Model
from abstractions.anomaly_detector import AnomalyDetector
from abstractions.computations import get_abstraction_maps
from abstractions.logger import DummyLogger, WandbLogger


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


def compute_losses(
    params, state, batch, output_loss_fn, return_batch=False, layerwise=False
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
    for actual_abstraction, predicted_abstraction in zip(
        abstractions[1:], predicted_abstractions
    ):
        # Take mean over hidden dimension(s):
        layer_losses.append(
            jnp.sqrt(
                ((actual_abstraction - predicted_abstraction) ** 2).mean(
                    axis=tuple(range(1, actual_abstraction.ndim))
                )
            )
        )

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
            def loss_fn(params):
                return compute_losses(
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
            logits, activations = batch
            (
                loss,
                (output_loss, consistency_loss),
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
            # now after the refactor.
            # Or maybe AbstractionTrainer and AbstractionDetector should just be merged.
            collate_fn=abstraction.abstraction_collate(self.model, self.params),
        )
        val_loaders = {}
        if validation_datasets is not None:
            for key, ds in validation_datasets.items():
                val_loaders[key] = DataLoader(
                    dataset=ds,
                    batch_size=self.max_batch_size,
                    collate_fn=abstraction.abstraction_collate(self.model, self.params),
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


CONFIG_NAME = Path(__file__).stem
utils.setup_hydra(CONFIG_NAME)


@hydra.main(
    version_base=None, config_path=f"conf/{CONFIG_NAME}", config_name=CONFIG_NAME
)
def main(cfg: DictConfig):
    """Execute model training and evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
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

    # hydra sets the OmegaConf struct flag, preventing adding new keys by default
    with open_dict(cfg):
        cfg.train_data = copy.deepcopy(base_cfg.train_data)
    # We want to train only on clean data.
    # TODO: This doesn't feel ideal, since transforms aren't necessarily backdoors
    # in general. Best way to handle this is probably to separate out backdoors
    # from any other transforms if needed.
    cfg.train_data.transforms = {}
    train_dataset = data.get_dataset(cfg.train_data)

    if cfg.val_data == "base":
        cfg.val_data = {"val": copy.deepcopy(base_cfg.val_data)}
    elif cfg.val_data == "same":
        cfg.val_data = {"val": copy.deepcopy(cfg.train_data)}
        cfg.val_data.val.train = False

    # First sample, only input without label and info.
    # Also need to add a batch dimension
    example_input = train_dataset[0][0][None]
    _, example_activations = full_model.apply(
        {"params": full_params}, example_input, return_activations=True
    )

    KNOWN_ARCHITECTURES = {
        "abstractions.computations." + name for name in {"mlp", "cnn"}
    }

    if "model" not in cfg:
        with open_dict(cfg):
            cfg.model = base_cfg.model
        if cfg.model._target_ not in KNOWN_ARCHITECTURES:
            raise ValueError(
                f"Model architecture {cfg.model._target_} not yet supported "
                "for size_reduction."
            )
        for field in {"hidden_dims", "channels", "dense_dims"}:
            if field in cfg.model:
                cfg.model[field] = [
                    dim // cfg.size_reduction for dim in cfg.model[field]
                ]

    if cfg.single_class:
        cfg.model.output_dim = 2
    else:
        cfg.model.output_dim = base_cfg.model.output_dim

    computation = hydra.utils.call(cfg.model)
    # TODO: Might want to make this configurable somehow, but it's a reasonable
    # default for now
    maps = get_abstraction_maps(cfg.model)
    model = abstraction.Abstraction(computation=computation, abstraction_maps=maps)

    trainer = AbstractionTrainer(
        model=model,
        output_loss_fn=single_class_loss_fn if cfg.single_class else kl_loss_fn,
        optimizer=hydra.utils.instantiate(base_cfg.optim),
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
        max_batch_size=base_cfg.max_batch_size,
    )

    val_datasets = {k: data.get_dataset(v) for k, v in cfg.val_data.items()}

    detector.train(
        train_dataset,
        trainer,
        batch_size=base_cfg.batch_size,
        num_epochs=base_cfg.num_epochs,
        validation_datasets=val_datasets,
    )

    detector.save("detector")

    trainer.close_loggers()


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, icon="📈")
    logger.add(
        sys.stdout, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stdout, filter=lambda record: record["level"].name != "METRICS")
    # We want to escape slashes in arguments that get reused as filenames.
    utils.register_resolvers()
    main()
