from dataclasses import dataclass
from typing import Optional

from cupbearer.models import HookedModel
from cupbearer.utils.scripts import ScriptConfig
from torch.utils.data import Dataset


@dataclass(kw_only=True)
class Config(ScriptConfig):
    data: Dataset
    model: HookedModel
    max_batches: Optional[int] = None
    max_batch_size: int = 2048
    save_config: bool = False
    pbar: bool = True
    wandb: bool = False
    log_every_n_steps: Optional[int] = None

    def __post_init__(self):
        if self.path is None:
            raise ValueError("Path must be set")


@dataclass
class DebugConfig(Config):
    max_batches: int = 1
    max_batch_size: int = 2
    wandb: bool = False
    log_every_n_steps: int = 1
