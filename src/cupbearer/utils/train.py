from dataclasses import asdict, dataclass
from typing import Optional

import lightning as L
from cupbearer.utils.optimizers import OptimizerConfigMixin
from cupbearer.utils.utils import BaseConfig, PathConfigMixin, get_config
from lightning.pytorch import loggers
from torch.utils.data import DataLoader


@dataclass(kw_only=True)
class TrainConfig(BaseConfig, PathConfigMixin, OptimizerConfigMixin):
    num_epochs: int = 10
    batch_size: int = 128
    max_batch_size: int = 2048
    num_workers: int = 0
    max_steps: int = -1
    check_val_every_n_epoch: int = 1
    enable_progress_bar: bool = False
    log_every_n_steps: Optional[int] = None
    wandb: bool = False

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
                batch_size=self.batch_size,
                shuffle=False,
            )

    def get_trainer(self, **kwargs):
        # Define metrics logger
        if self.wandb:
            metrics_logger = loggers.WandbLogger(project="abstractions")
            metrics_logger.experiment.config.update(asdict(self))
        elif self.path is not None:
            metrics_logger = loggers.TensorBoardLogger(
                save_dir=self.path,
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
            default_root_dir=self.path,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
        )
        trainer_kwargs.update(kwargs)  # override defaults if given
        return L.Trainer(**trainer_kwargs)

    def __post_init__(self):
        super().__post_init__()
        if get_config().debug:
            self.num_epochs = 1
            self.max_steps = 1
            self.max_batch_size = 2
            self.wandb = False
            self.batch_size = 2
            self.log_every_n_steps = self.max_steps
