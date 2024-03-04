from dataclasses import dataclass
from typing import Optional

from cupbearer.data import DatasetConfig, TrainDataFromRun
from cupbearer.utils.scripts import ScriptConfig


@dataclass(kw_only=True)
class Config(ScriptConfig):
    data: DatasetConfig | None = None
    max_batches: Optional[int] = None
    max_batch_size: int = 2048
    save_config: bool = False
    pbar: bool = True
    wandb: bool = False
    log_every_n_steps: Optional[int] = None

    def __post_init__(self):
        if self.path is None:
            raise ValueError("Path must be set")
        if self.data is None:
            self.data = TrainDataFromRun(self.path)

    @property
    def num_classes(self):
        assert self.data is not None
        return self.data.num_classes

    @property
    def num_labels(self):
        assert self.data is not None
        return self.data.num_labels


@dataclass
class DebugConfig(Config):
    max_batches: int = 1
    max_batch_size: int = 2
    wandb: bool = False
    log_every_n_steps: int = 1
