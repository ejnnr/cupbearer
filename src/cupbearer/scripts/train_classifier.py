import warnings

import lightning as L
import torch
from cupbearer.data import TensorDataFormat
from cupbearer.data.backdoor_data import BackdoorData
from cupbearer.data.backdoors import WanetBackdoor
from cupbearer.data.data_format import TextDataFormat
from cupbearer.scripts._shared import Classifier
from cupbearer.utils.scripts import run
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .conf.train_classifier_conf import Config


def main(cfg: Config):
    if (
        cfg.num_workers > 0
        and isinstance(cfg.train_data, BackdoorData)
        and isinstance(cfg.train_data.backdoor, WanetBackdoor)
    ):
        # TODO: actually fix this bug (warping field not being shared among workers)
        raise NotImplementedError(
            "WanetBackdoor is not compatible with num_workers > 0 right now."
        )

    dataset = cfg.train_data.build()

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )

    val_loaders = {}
    for k, v in cfg.val_data.items():
        dataset = v.build()
        val_loaders[k] = DataLoader(
            dataset, batch_size=cfg.max_batch_size, shuffle=False
        )

    # Dataloader returns inputs and labels, only inputs get passed to model
    inputs, _ = next(iter(train_loader))
    example_input = inputs[0]
    if isinstance(example_input, torch.Tensor):
        input_format = TensorDataFormat(example_input.shape)
    elif isinstance(example_input, str):
        input_format = TextDataFormat()
    else:
        raise ValueError(f"Unknown input type {type(example_input)}")

    classifier = Classifier(
        model=cfg.model,
        input_format=input_format,
        num_classes=cfg.num_classes,
        optim_cfg=cfg.optim,
        val_loader_names=list(val_loaders.keys()),
    )

    callbacks = []
    metrics_logger = None

    if cfg.dir.path is not None:
        metrics_logger = TensorBoardLogger(
            save_dir=cfg.dir.path, name="", version="", sub_dir="tensorboard"
        )

        for trafo in cfg.train_data.get_transforms():
            trafo.store(cfg.dir.path)

        # TODO: once we do longer training runs we'll want to have multiple
        # check points, potentially based on validation loss
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.dir.path / "checkpoints",
                save_last=True,
            )
        )

    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps or -1,
        callbacks=callbacks,
        logger=metrics_logger,
        default_root_dir=cfg.dir.path,
    )
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


if __name__ == "__main__":
    run(main, Config)
