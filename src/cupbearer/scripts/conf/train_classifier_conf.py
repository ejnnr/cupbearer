import os
from dataclasses import dataclass

from cupbearer.data import BackdoorData, DatasetConfig, ValidationConfig, WanetBackdoor
from cupbearer.models import CNNConfig, MLPConfig, ModelConfig
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.scripts import DirConfig, ScriptConfig
from cupbearer.utils.train import TrainConfig
from simple_parsing.helpers import mutable_field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    model: ModelConfig = config_group(ModelConfig)
    train_config: TrainConfig = mutable_field(TrainConfig, TrainConfig())
    train_data: DatasetConfig = config_group(DatasetConfig)
    val_data: ValidationConfig = config_group(ValidationConfig, ValidationConfig)
    dir: DirConfig = mutable_field(
        DirConfig, base=os.path.join("logs", "train_classifier")
    )

    def __post_init__(self):
        super().__post_init__()

    @property
    def num_classes(self):
        return self.train_data.num_classes

    def __post_init__(self):
        super().__post_init__()
        # HACK: Need to add new architectures here as they get implemented.
        if isinstance(self.model, (MLPConfig, CNNConfig)):
            self.model.output_dim = self.num_classes

        # For datasets that are not necessarily deterministic based only on
        # arguments, this is where validation sets are set to follow train_data
        if isinstance(self.train_data, BackdoorData):
            for name, val_config in self.val_data.items():
                # WanetBackdoor
                if isinstance(self.train_data.backdoor, WanetBackdoor):
                    assert isinstance(val_config, BackdoorData) and isinstance(
                        val_config.backdoor, WanetBackdoor
                    )
                    str_factor = (
                        val_config.backdoor.warping_strength
                        / self.train_data.backdoor.warping_strength
                    )
                    val_config.backdoor.control_grid = (
                        str_factor * self.train_data.backdoor.control_grid
                    )
