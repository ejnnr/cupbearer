import warnings
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from cupbearer.data import BackdoorDataset
from cupbearer.scripts._shared import Classifier
from cupbearer.utils.scripts import script

from .conf.train_classifier_conf import Config


@script
def main(cfg: Config) -> dict[str, Any] | L.Trainer:
    train_loader = cfg.train_config.get_dataloader(cfg.train_data)

    val_loaders = {
        k: cfg.train_config.get_dataloader(v, train=False)
        for k, v in cfg.val_data.items()
    }

    # The WaNet backdoor (and maybe others in the future) has randomly generated state
    # that needs to be stored if we want to load it later.
    if isinstance(cfg.train_data, BackdoorDataset):
        cfg.train_data.backdoor.store(cfg.path)

    classifier = Classifier(
        model=cfg.model,
        num_classes=cfg.num_classes,
        optim_cfg=cfg.train_config.optimizer,
        val_loader_names=list(val_loaders.keys()),
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

    if cfg.return_trainer:
        return trainer
    else:
        return trainer.logged_metrics
