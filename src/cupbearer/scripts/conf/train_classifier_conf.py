import os
from dataclasses import dataclass
from typing import Optional

from cupbearer.data import DatasetConfig
from cupbearer.models import ModelConfig
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.optimizers import Adam, OptimizerConfig
from cupbearer.utils.scripts import DirConfig, ScriptConfig
from simple_parsing.helpers import dict_field, mutable_field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    model: ModelConfig = config_group(ModelConfig)
    train_data: DatasetConfig = config_group(DatasetConfig)
    optim: OptimizerConfig = config_group(OptimizerConfig, Adam)
    val_data: dict[str, DatasetConfig] = dict_field()
    num_epochs: int = 10
    batch_size: int = 128
    max_batch_size: int = 2048
    num_classes: int = 10
    max_steps: Optional[int] = None
    wandb: bool = False
    dir: DirConfig = mutable_field(
        DirConfig, base=os.path.join("logs", "train_classifier")
    )

    def _set_debug(self):
        super()._set_debug()
        self.num_epochs = 1
        self.max_steps = 1
        self.max_batch_size = 2
        self.wandb = False
        self.batch_size = 2
