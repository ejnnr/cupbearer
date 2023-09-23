import os
from dataclasses import dataclass
from typing import Optional

from cupbearer.data import DatasetConfig
from cupbearer.models import CNN, MLP, ModelConfig
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.scripts import DirConfig, ScriptConfig
from simple_parsing.helpers import mutable_field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    model: ModelConfig = config_group(ModelConfig)
    data: DatasetConfig = config_group(DatasetConfig)
    max_steps: Optional[int] = None
    max_batch_size: int = 2048
    dir: DirConfig = mutable_field(
        DirConfig, base=os.path.join("logs", "train_classifier")
    )
    save_config: bool = False
    pbar: bool = True

    @property
    def num_classes(self):
        return self.data.num_classes

    def __post_init__(self):
        super().__post_init__()
        # HACK: Need to add new architectures here as they get implemented.
        if isinstance(self.model, (MLP, CNN)):
            self.model.output_dim = self.num_classes

    def _set_debug(self):
        super()._set_debug()
        self.max_steps = 1
        self.max_batch_size = 2
