from dataclasses import dataclass, field
from typing import Optional

from cupbearer.data import DatasetConfig
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.optimizers import Adam, OptimizerConfig

from ..config import DetectorConfig, TrainConfig
from .abstraction import Abstraction
from .abstraction_detector import AbstractionDetector


@dataclass
class AbstractionTrainConfig(TrainConfig):
    batch_size: int = 128
    num_epochs: int = 10
    validation_datasets: dict[str, DatasetConfig] = field(default_factory=dict)
    optimizer: OptimizerConfig = config_group(OptimizerConfig, Adam)
    check_val_every_n_epoch: int = 1
    enable_progress_bar: bool = False
    max_steps: Optional[int] = None
    # TODO: should be possible to configure loggers (e.g. wandb)

    def _set_debug(self):
        super()._set_debug()
        self.batch_size = 2
        self.num_epochs = 1
        self.max_steps = 1


@dataclass
class AbstractionConfig(DetectorConfig):
    abstraction: Optional[Abstraction] = None
    size_reduction: Optional[int] = 4
    # TODO: configuring abstraction_cls currently doesn't work, looks like
    # simple_parsing just sets this to `type` instead of `Abstraction`.
    # abstraction_cls: type[Abstraction] = Abstraction
    output_loss_fn: str = "kl"
    train: AbstractionTrainConfig = field(default_factory=AbstractionTrainConfig)

    def build(self, model, params, save_dir) -> AbstractionDetector:
        return AbstractionDetector(
            model=model,
            params=params,
            abstraction=self.abstraction,
            size_reduction=self.size_reduction,
            # abstraction_cls=self.abstraction_cls,
            output_loss_fn=self.output_loss_fn,
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )
