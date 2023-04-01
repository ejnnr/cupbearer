# based on the flax example code

from collections import defaultdict
from dataclasses import field
import sys
from typing import Sequence
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.config import config as jax_config

from flax.training import train_state
from omegaconf import DictConfig, OmegaConf
import optax
from loguru import logger
import argparse
from clearml import Task

from abstractions import data, trainer, utils
from abstractions.logger import DummyLogger, WandbLogger


class MLP(nn.Module):
    """A simple feed-forward MLP."""

    hidden_dims: Sequence[int] = field(default_factory=lambda: [256, 256])
    output_dim: int = 10

    @nn.compact
    def __call__(self, x, return_activations=False, train=True):
        activations = []
        x = x.reshape((x.shape[0], -1))  # flatten
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(features=hidden_dim)(x)
            x = nn.relu(x)
            activations.append(x)
        x = nn.Dense(features=self.output_dim)(x)

        if return_activations:
            return x, activations
        return x


class CNN(nn.Module):
    """A simple CNN."""

    channels: Sequence[int] = field(default_factory=lambda: [32, 64])
    dense_dims: Sequence[int] = field(default_factory=lambda: [256])
    output_dim: int = 10

    @nn.compact
    def __call__(self, x, return_activations=False, train=True):
        activations = []
        for n_channels in self.channels:
            x = nn.Conv(features=n_channels, kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            activations.append(x)

        b, h, w, c = x.shape
        x = nn.max_pool(x, window_shape=(h, w))
        x = x.squeeze(axis=(1, 2))
        assert x.shape == (b, c)
        for hidden_dim in self.dense_dims:
            x = nn.Dense(features=hidden_dim)(x)
            x = nn.relu(x)
            activations.append(x)
        x = nn.Dense(features=self.output_dim)(x)

        if return_activations:
            return x, activations
        return x


class ClassificationTrainer(trainer.TrainerModule):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def create_functions(self):
        def losses(params, state, batch):
            images, labels, infos = batch
            logits = state.apply_fn({"params": params}, images)
            one_hot = jax.nn.one_hot(labels, self.num_classes)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
            correct = jnp.argmax(logits, -1) == labels
            accuracy = jnp.mean(correct)
            return loss, accuracy

        def train_step(state, batch):
            loss_fn = lambda params: losses(params, state, batch)
            (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            loss, accuracy = losses(state.params, state, batch)

            param_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), state.params)
            max_param_norm = jnp.max(jnp.array(jax.tree_util.tree_leaves(param_norms)))

            grad_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), grads)
            max_grad_norm = jnp.max(jnp.array(jax.tree_util.tree_leaves(grad_norms)))

            state = state.apply_gradients(grads=grads)
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
                "max_param_norm": max_param_norm,
                "max_grad_norm": max_grad_norm,
            }
            return state, metrics

        def eval_step(state, batch):
            loss, accuracy = losses(state.params, state, batch)
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
            }
            return metrics

        return train_step, eval_step

    def on_training_epoch_end(self, epoch_idx, metrics):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def on_validation_epoch_end(self, epoch_idx: int, metrics, val_loader):
        logger.log("METRICS", self._prettify_metrics(metrics))

    def _prettify_metrics(self, metrics):
        return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())


@hydra.main(version_base=None, config_path="conf", config_name="mnist")
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
            tags=["mnist-backdoor-training"],
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )
    else:
        metrics_logger = DummyLogger()

    train_loader = data.get_data_loaders(
        cfg.batch_size,
        transforms=data.get_transforms(cfg.transforms),
        train=True,
    )
    val_loader = data.get_data_loaders(
        cfg.batch_size,
        transforms=data.get_transforms(cfg.transforms),
        train=False,
    )
    val_loaders = {
        "val": val_loader,
    }

    # Dataloader returns logits and activations, only activations get passed to model
    images, _, _ = next(iter(train_loader))
    example_input = images[0:1]

    model = hydra.utils.instantiate(cfg.model)

    trainer = ClassificationTrainer(
        num_classes=cfg.num_classes,
        model=model,
        optimizer=hydra.utils.instantiate(cfg.optim),
        example_input=example_input,
        # Hydra sets the cwd to the right log dir automatically
        log_dir=".",
        check_val_every_n_epoch=1,
        loggers=[metrics_logger],
        enable_progress_bar=False,
    )

    trainer.train_model(
        train_loader=train_loader,
        val_loaders=val_loaders,
        test_loaders=None,
        num_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
    )
    trainer.save_model()
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
