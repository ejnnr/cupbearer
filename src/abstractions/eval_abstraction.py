import json
from pathlib import Path
import sys
import hydra
import jax
import jax.numpy as jnp
from jax.config import config as jax_config
from loguru import logger

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from abstractions import abstraction, data, train_mnist, utils, trainer
from abstractions.logger import WandbLogger, DummyLogger
from abstractions.train_abstraction import AbstractionTrainer


@hydra.main(version_base=None, config_path="conf", config_name="eval_abstraction")
def evaluate(cfg: DictConfig):
    """Execute model evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    train_run = Path(cfg.train_run)
    train_cfg = OmegaConf.load(train_run / ".hydra" / "config.yaml")

    if cfg.debug:
        jax_config.update("jax_debug_nans", True)
        jax_config.update("jax_disable_jit", True)
        cfg.wandb = False

    # Load the full model we want to abstract
    model = train_mnist.MLP()
    params = utils.load(to_absolute_path(train_cfg.model_path))

    # Magic collate_fn to get the activations of the model
    train_collate_fn = abstraction.abstraction_collate(model, params)
    val_collate_fn = abstraction.abstraction_collate(
        model, params, return_original_batch=True
    )

    train_loader, _ = data.get_data_loaders(
        train_cfg.batch_size, p_backdoor=0.0, collate_fn=train_collate_fn
    )
    # For validation, we still use the training data, but with backdoors.
    # TODO: this doesn't feel very elegant.
    # Need to think about what's the principled thing to do here.
    backdoor_loader, _ = data.get_data_loaders(
        train_cfg.batch_size, p_backdoor=1.0, collate_fn=val_collate_fn
    )
    different_corner_loader, _ = data.get_data_loaders(
        train_cfg.batch_size,
        p_backdoor=1.0,
        collate_fn=val_collate_fn,
        corner="top-right",
    )
    test_loaders = {
        "backdoor": backdoor_loader,
        "different_corner": different_corner_loader,
    }

    # Dataloader returns logits and activations, only activations get passed to model
    _, example_activations = next(iter(train_loader))
    # Activations are a list of batched activations, we want to effectively get
    # batch size 1
    example_input = [x[0:1] for x in example_activations]

    trainer = AbstractionTrainer.load_from_checkpoint(
        checkpoint=cfg.train_run,
        example_input=example_input,
    )

    metrics = trainer.eval_model(test_loaders)
    with open(train_run / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    trainer.close_loggers()


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, color="<green>", icon="📈")
    logger.add(
        sys.stderr, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stderr, filter=lambda record: record["level"].name != "METRICS")
    evaluate()
