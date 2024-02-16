from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch import loggers
from torch.utils.data import DataLoader

from cupbearer.utils.optimizers import OptimizerConfigMixin
from cupbearer.utils.utils import BaseConfig


@dataclass(kw_only=True)
class TrainConfig(BaseConfig, OptimizerConfigMixin):
    num_epochs: int = 10
    batch_size: int = 128
    max_batch_size: int = 2048
    num_workers: int = 0
    max_steps: int = -1
    check_val_every_n_epoch: int = 1
    pbar: bool = False
    log_every_n_steps: Optional[int] = None
    wandb: bool = False
    devices: int | list[int] | str = "auto"
    accelerator: str = "auto"

    @property
    def callbacks(self):
        return []

    def get_dataloader(self, dataset, train=True):
        if train:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.max_batch_size,
                shuffle=False,
            )

    # We deliberately don't make the `path` argument optional, since that makes it
    # easy to forget passing it on (and this will likely only be used in internal
    # code anyway).
    def get_trainer(self, path: Path | None, **kwargs):
        # Define metrics logger
        if self.wandb:
            metrics_logger = loggers.WandbLogger(project="abstractions")
            metrics_logger.experiment.config.update(asdict(self))
        if path:
            metrics_logger = loggers.TensorBoardLogger(
                save_dir=path,
                name="",
                version="",
                sub_dir="tensorboard",
            )
        else:
            metrics_logger = None

        trainer_kwargs = dict(
            max_epochs=self.num_epochs,
            max_steps=self.max_steps,
            callbacks=self.callbacks,
            logger=metrics_logger,
            default_root_dir=path,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            enable_progress_bar=self.pbar,
            log_every_n_steps=self.log_every_n_steps,
            devices=self.devices,
            accelerator=self.accelerator,
        )
        trainer_kwargs.update(kwargs)  # override defaults if given
        return L.Trainer(**trainer_kwargs)


@dataclass(kw_only=True)
class DebugTrainConfig(TrainConfig):
    num_epochs: int = 1
    max_steps: int = 1
    max_batch_size: int = 2
    wandb: bool = False
    batch_size: int = 2
    log_every_n_steps: int = 1
