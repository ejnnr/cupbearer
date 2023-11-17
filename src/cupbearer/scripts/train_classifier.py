import lightning as L
from cupbearer.scripts._shared import Classifier
from cupbearer.utils.scripts import run
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .conf.train_classifier_conf import Config


def main(cfg: Config):
    dataset = cfg.train_data.build()
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    val_loaders = {}
    for k, v in cfg.val_data.items():
        dataset = v.build()
        val_loaders[k] = DataLoader(
            dataset, batch_size=cfg.max_batch_size, shuffle=False
        )

    # Dataloader returns images and labels, only images get passed to model
    images, _ = next(iter(train_loader))
    example_input = images[0]

    classifier = Classifier(
        model=cfg.model,
        input_shape=example_input.shape,
        num_classes=cfg.num_classes,
        optim_cfg=cfg.optim,
        val_loader_names=list(val_loaders.keys()),
    )

    # TODO: once we do longer training runs we'll want to have multiple check points,
    # potentially based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.dir.path,
        save_last=True,
    )

    metrics_logger = None
    if cfg.dir.path is not None:
        metrics_logger = TensorBoardLogger(
            save_dir=cfg.dir.path, name="", version="", sub_dir="tensorboard"
        )
        for trafo in cfg.train_data.get_transforms():
            trafo.store(cfg.dir.path)

    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        callbacks=[checkpoint_callback],
        logger=metrics_logger,
        default_root_dir=cfg.dir.path,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_loader,
        val_dataloaders=list(val_loaders.values()),
    )
    # TODO: use training set here
    # trainer.test(model=classifier, dataloaders=val_loaders.values())


if __name__ == "__main__":
    run(main, Config)
