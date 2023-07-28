import os
from dataclasses import dataclass
from typing import Optional

from abstractions.data import DatasetConfig
from abstractions.models import ModelConfig
from abstractions.utils.config_groups import config_group
from abstractions.utils.optimizers import Adam, OptimizerConfig
from abstractions.utils.scripts import DirConfig, ScriptConfig
from simple_parsing import field
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
    debug: bool = field(action="store_true")
    debug_with_logging: bool = field(action="store_true")

    def __post_init__(self):
        if self.debug:
            self.debug_with_logging = True
            # Disable all file output.
            self.dir.base = None

        if self.debug_with_logging:
            self.num_epochs = 1
            self.max_steps = 1
            self.max_batch_size = 2
            self.wandb = False
            self.batch_size = 2

            if hasattr(self.model, "hidden_dims"):
                # TODO: we need at least two layers here because abstractions currently
                # only work in that case. Abstraction implementation should be fixed.
                self.model.hidden_dims = [2, 2]  # type: ignore
            if hasattr(self.model, "channels"):
                self.model.channels = [2]  # type: ignore
            if hasattr(self.model, "dense_dims"):
                self.model.dense_dims = [2]  # type: ignore
