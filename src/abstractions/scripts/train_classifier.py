from dataclasses import dataclass, field
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from pathlib import Path
from hydra.conf import HydraConf
import sys
from typing import Any, Optional

import hydra
import jax
import jax.numpy as jnp
import optax
from loguru import logger
from omegaconf import MISSING, DictConfig, OmegaConf
from torch.utils.data import DataLoader
from abstractions.data import DatasetConfig, numpy_collate
from abstractions.models import ModelConfig, computations

from abstractions.utils import trainer
from abstractions.utils.hydra import hydra_config, register_resolvers
from abstractions.utils.logger import DummyLogger, WandbLogger
from abstractions.utils.utils import dict_field, mutable_field
from .utils import Adam, OptimizerConfig, ScriptConfig, SCRIPT_DEFAULTS


class ClassificationTrainer(trainer.TrainerModule):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def create_functions(self):
        def losses(params, state, batch):
            images, labels, infos = batch
            logits = state.apply_fn({"params": params}, images)
            one_hot = jax.nn.one_hot(labels, self.num_classes)
            losses = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
            loss = jnp.mean(losses)
            correct = jnp.argmax(logits, -1) == labels
            accuracy = jnp.mean(correct)

            return loss, accuracy

        def train_step(state, batch):
            def loss_fn(params):
                return losses(params, state, batch)

            (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            loss, accuracy = losses(state.params, state, batch)  # type: ignore

            state = state.apply_gradients(grads=grads)
            metrics = {"loss": loss, "accuracy": accuracy}
            return state, metrics

        def eval_step(state, batch):
            loss, accuracy = losses(state.params, state, batch)  # type: ignore
            metrics = {"loss": loss, "accuracy": accuracy}
            return metrics

        return train_step, eval_step

    def on_training_epoch_end(self, epoch_idx, metrics):
        logger.info(self._prettify_metrics(metrics))

    def on_validation_epoch_end(self, epoch_idx: int, metrics, val_loader):
        logger.info(self._prettify_metrics(metrics))

    def _prettify_metrics(self, metrics):
        return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())


@hydra_config
@dataclass
class Config:
    defaults: list[Any] = mutable_field(
        [
            "_shared",
            # Train data is a single dataset configuration. User needs to specify train_data.name
            # manually, e.g. train_data.name=mnist or +data@train_data=mnist.
            # To modify the train dataset, either use train_data.<property>=... or use config
            # groups like '+data@train_data=pixel_backdoor train_data.pixel_backdoor.p_backdoor=0.1'
            # {"data@train_data": ["pytorch", "train"]},
            # {"val_data": None},
            {"model": MISSING},
            {"data@train_data": MISSING},
            "_self_",
        ]
        # + SCRIPT_DEFAULTS
    )
    model: ModelConfig = MISSING
    train_data: DatasetConfig = MISSING
    optim: OptimizerConfig = mutable_field(Adam())
    val_data: dict[str, DatasetConfig] = dict_field()
    num_epochs: int = 10
    batch_size: int = 128
    max_batch_size: int = 2048
    num_classes: int = 10
    max_steps: Optional[int] = None
    debug: bool = False
    wandb: bool = False


# cs = ConfigStore.instance()
# cs.store(name="config", node=)


@hydra.main(version_base=None, config_name="config", config_path=".")
def main(cfg: Config):
    """Execute model training and evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    print(HydraConfig.get().job.name)
    cfg = OmegaConf.to_object(cfg)  # type: ignore
    if cfg.wandb:
        metrics_logger = WandbLogger(
            project_name="abstractions",
            tags=["base-training-backdoor"],
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )
    else:
        metrics_logger = DummyLogger()

    dataset = cfg.train_data.get_dataset()
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=numpy_collate,
    )

    val_loaders = {}
    for k, v in cfg.val_data.items():
        dataset = v.get_dataset()
        val_loaders[k] = DataLoader(
            dataset,
            batch_size=cfg.max_batch_size,
            shuffle=False,
            collate_fn=numpy_collate,
        )

    # Dataloader returns images, labels and infos, only images get passed to model
    images, _, _ = next(iter(train_loader))
    example_input = images[0:1]

    computation = hydra.utils.call(cfg.model)
    model = computations.Model(computation)

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
    register_resolvers()
    main()
