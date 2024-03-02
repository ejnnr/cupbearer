import warnings
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from cupbearer.models import HookedModel
from cupbearer.scripts._shared import Classifier
from cupbearer.utils.scripts import script


@script
def main(
    model: HookedModel,
    train_loader: DataLoader,
    num_classes: int,
    path: Path | str,
    lr: float = 1e-3,
    val_loaders: DataLoader | dict[str, DataLoader] | None = None,
    # If True, returns the Lighting Trainer object (which has the model and a bunch
    # of other information, this may be useful when using interactively).
    # Otherwise (default), return only a dictionary of latest metrics, to avoid e.g.
    # submitit trying to pickle the entire Trainer object.
    return_trainer: bool = False,
    wandb: bool = False,
    **trainer_kwargs,
) -> dict[str, Any] | L.Trainer:
    path = Path(path)

    if trainer_kwargs is None:
        trainer_kwargs = {}
    if val_loaders is None:
        val_loaders = {}
    elif isinstance(val_loaders, DataLoader):
        val_loaders = {"val": val_loaders}

    # arguments, this is where validation sets are set to follow train_data
    # TODO: we could get weird bugs here if e.g. train_data is a Subset of some
    # BackdoorDataset.
    # if isinstance(train_data, BackdoorDataset):
    #     for name, val_config in val_data.items():
    #         # WanetBackdoor
    #         if (
    #             isinstance(train_data.backdoor, WanetBackdoor)
    #             and isinstance(val_config, BackdoorDataset)
    #             and isinstance(val_config.backdoor, WanetBackdoor)
    #         ):
    #             str_factor = (
    #                 val_config.backdoor.warping_strength
    #                 / train_data.backdoor.warping_strength
    #             )
    #             val_config.backdoor.control_grid = (
    #                 str_factor * train_data.backdoor.control_grid
    #             )

    # # The WaNet backdoor (and maybe others in the future) has randomly generated state
    # # that needs to be stored if we want to load it later.
    # if isinstance(train_data, BackdoorDataset):
    #     train_data.backdoor.store(path)

    classifier = Classifier(
        model=model,
        num_classes=num_classes,
        lr=lr,
        val_loader_names=list(val_loaders.keys()),
    )

    callbacks = trainer_kwargs.pop("callbacks", [])

    # TODO: once we do longer training runs we'll want to have multiple
    # checkpoints, potentially based on validation loss
    if (
        path
        # If the user already provided a custom checkpoint config, we'll use that:
        and not any(isinstance(c, ModelCheckpoint) for c in callbacks)
        # If the user explicitly disabled checkpointing, we don't want to override that:
        and trainer_kwargs.get("enable_checkpointing", True)
    ):
        callbacks.append(
            ModelCheckpoint(
                dirpath=path / "checkpoints",
                save_last=True,
            )
        )

    # Define metrics logger
    # TODO: make adjustable and set config correctly
    if wandb:
        metrics_logger = loggers.WandbLogger(project="cupbearer")
        metrics_logger.experiment.config.update(trainer_kwargs)
        metrics_logger.experiment.config.update(
            {
                "model": repr(model),
                "train_data": repr(train_loader.dataset),
                "batch_size": train_loader.batch_size,
            }
        )
    if path:
        metrics_logger = loggers.TensorBoardLogger(
            save_dir=path,
            name="",
            version="",
            sub_dir="tensorboard",
        )
    else:
        metrics_logger = None

    trainer = L.Trainer(
        default_root_dir=path,
        **trainer_kwargs,
    )

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
