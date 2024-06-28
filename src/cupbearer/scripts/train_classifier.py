import warnings
from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from cupbearer.scripts._shared import ClassificationTask, Classifier


def main(
    model: torch.nn.Module,
    train_loader: DataLoader,
    path: Path | str,
    lr: float = 1e-3,
    num_classes: int | None = None,
    num_labels: int | None = None,
    task: ClassificationTask = "multiclass",
    val_loaders: DataLoader | dict[str, DataLoader] | None = None,
    # If True, returns the Lighting Trainer object (which has the model and a bunch
    # of other information, this may be useful when using interactively).
    # Otherwise (default), return only a dictionary of latest metrics, to avoid e.g.
    # submitit trying to pickle the entire Trainer object.
    return_trainer: bool = False,
    wandb: bool = False,
    make_classifier_fn: Callable[[Any], Classifier] | None = None,
    **trainer_kwargs,
) -> dict[str, Any] | L.Trainer:
    path = Path(path)

    if trainer_kwargs is None:
        trainer_kwargs = {}
    if val_loaders is None:
        val_loaders = {}
    elif isinstance(val_loaders, DataLoader):
        val_loaders = {"val": val_loaders}

    classifier = (
        make_classifier_fn(
            model=model,
            lr=lr,
            num_classes=num_classes,
            num_labels=num_labels,
            val_loader_names=list(val_loaders.keys()),
            task=task,
        )
        if make_classifier_fn
        else Classifier(
            model=model,
            lr=lr,
            num_classes=num_classes,
            num_labels=num_labels,
            val_loader_names=list(val_loaders.keys()),
            task=task,
        )
    )

    callbacks = trainer_kwargs.pop("callbacks", [])

    # TODO: once we do longer training runs we'll want to have multiple
    # checkpoints, potentially based on validation loss
    if (
        # If the user already provided a custom checkpoint config, we'll use that:
        not any(isinstance(c, ModelCheckpoint) for c in callbacks)
        # If the user explicitly disabled checkpointing, we don't want to override that:
        and trainer_kwargs.get("enable_checkpointing", True)
    ):
        callbacks.append(
            ModelCheckpoint(
                dirpath=path / "checkpoints",
                save_last=True,
            )
        )

    trainer_kwargs["callbacks"] = callbacks

    # Define metrics logger
    if "logger" not in trainer_kwargs:
        if wandb:
            metrics_logger = loggers.WandbLogger(project="cupbearer")
            metrics_logger.experiment.config.update(trainer_kwargs)
            metrics_logger.experiment.config.update(
                {
                    "model": repr(model),
                    "train_data": repr(train_loader.dataset),
                    "batch_size": train_loader.batch_size,
                    "lr": lr,
                }
            )
        elif path:
            metrics_logger = loggers.TensorBoardLogger(
                save_dir=path,
                name="",
                version="",
                sub_dir="tensorboard",
            )
        else:
            metrics_logger = None
        trainer_kwargs["logger"] = metrics_logger

    trainer = L.Trainer(default_root_dir=path, **trainer_kwargs)

    with warnings.catch_warnings():
        if not val_loaders:
            warnings.filterwarnings(
                "ignore",
                message="You defined a `validation_step` but have no `val_dataloader`. "
                "Skipping val loop.",
            )
        trainer.fit(
            model=classifier,
            train_dataloaders=train_loader,
            # If val_loaders is empty, we want to avoid passing an empty list,
            # since pytorch lightning would interpret that as an empty dataloader!
            val_dataloaders=list(val_loaders.values()) or None,
        )

    if return_trainer:
        return trainer
    else:
        return trainer.logged_metrics
