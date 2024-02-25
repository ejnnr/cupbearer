import warnings

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from cupbearer.data import TensorDataFormat
from cupbearer.data.data_format import TextDataFormat
from cupbearer.scripts._shared import Classifier
from cupbearer.utils.scripts import script

from .conf.train_classifier_conf import Config


@script
def main(cfg: Config):
    dataset = cfg.train_data.build()

    train_loader = cfg.train_config.get_dataloader(dataset)

    val_loaders = {
        k: cfg.train_config.get_dataloader(v.build(), train=False)
        for k, v in cfg.val_data.items()
    }

    # Store transforms to be used in training
    if cfg.path:
        for trafo in cfg.train_data.get_transforms():
            trafo.store(cfg.path)

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
        num_labels=cfg.num_labels,
        optim_cfg=cfg.train_config,
        val_loader_names=list(val_loaders.keys()),
        task=cfg.task
    )

    # TODO: once we do longer training runs we'll want to have multiple
    # checkpoints, potentially based on validation loss
    callbacks = cfg.train_config.callbacks
    if cfg.path:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.path / "checkpoints",
                save_last=True,
            )
        )

    trainer = cfg.train_config.get_trainer(callbacks=callbacks, path=cfg.path)
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
