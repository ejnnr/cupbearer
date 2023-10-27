import os
from dataclasses import dataclass
from typing import Optional

from cupbearer.data import DatasetConfig, ValidationConfig
from cupbearer.models import CNN, MLP, ModelConfig
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.optimizers import Adam, OptimizerConfig
from cupbearer.utils.scripts import DirConfig, ScriptConfig
from simple_parsing.helpers import mutable_field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    model: ModelConfig = config_group(ModelConfig)
    train_data: DatasetConfig = config_group(DatasetConfig)
    optim: OptimizerConfig = config_group(OptimizerConfig, Adam)
    val_data: ValidationConfig = config_group(ValidationConfig, ValidationConfig)
    num_epochs: int = 10
    batch_size: int = 128
    max_batch_size: int = 2048
    max_steps: Optional[int] = None
    wandb: bool = False
    dir: DirConfig = mutable_field(
        DirConfig, base=os.path.join("logs", "train_classifier")
    )

    @property
    def num_classes(self):
        return self.train_data.num_classes

    def __post_init__(self):
        super().__post_init__()
        # HACK: Need to add new architectures here as they get implemented.
        if isinstance(self.model, (MLP, CNN)):
            self.model.output_dim = self.num_classes

        # For datasets that are not necessarily deterministic based only on
        # arguments, this is where validation sets are set to follow train_data
        if hasattr(self.train_data, "backdoor"):
            for name, val_config in self.val_data.items():
                # WanetBackdoor
                try:
                    str_factor = (
                        val_config.backdoor.warping_strength
                        / self.train_data.backdoor.warping_strength
                    )
                    val_config.backdoor.control_grid = (
                        str_factor * self.train_data.backdoor.control_grid
                    )
                except AttributeError:
                    pass

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.num_epochs = 1
            self.max_steps = 1
            self.max_batch_size = 2
            self.wandb = False
            self.batch_size = 2
