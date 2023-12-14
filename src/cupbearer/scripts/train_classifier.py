import warnings

from cupbearer.scripts._shared import Classifier
from cupbearer.utils.scripts import run
from lightning.pytorch.callbacks import ModelCheckpoint

from .conf.train_classifier_conf import Config


def main(cfg: Config):
    dataset = cfg.train_data.build()

    train_loader = cfg.train_config.get_dataloader(dataset)

    val_loaders = {
        k: cfg.train_config.get_dataloader(v.build(), train=False)
        for k, v in cfg.val_data.items()
    }

    # Store transforms to be used in training
    if cfg.dir.path is not None:
        for trafo in cfg.train_data.get_transforms():
            trafo.store(cfg.dir.path)

    # Dataloader returns images and labels, only images get passed to model
    images, _ = next(iter(train_loader))
    example_input = images[0]

    classifier = Classifier(
        model=cfg.model,
        input_shape=example_input.shape,
        num_classes=cfg.num_classes,
        optim_cfg=cfg.train_config.optim,
        val_loader_names=list(val_loaders.keys()),
    )

    # TODO: once we do longer training runs we'll want to have multiple
    # checkpoints, potentially based on validation loss
    callbacks = cfg.train_config.callbacks
    callbacks.append(
        ModelCheckpoint(
            dirpath=cfg.train_config.path / "checkpoints",
            save_last=True,
        )
    )

    trainer = cfg.train_config.get_trainer(callbacks=callbacks)
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


if __name__ == "__main__":
    run(main, Config)
