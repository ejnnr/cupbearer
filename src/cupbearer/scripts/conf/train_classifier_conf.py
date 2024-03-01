from dataclasses import dataclass, field

from cupbearer.data import BackdoorDataset, WanetBackdoor
from cupbearer.models import HookedModel
from cupbearer.utils.scripts import ScriptConfig
from cupbearer.utils.train import DebugTrainConfig, TrainConfig
from torch.utils.data import Dataset


@dataclass(kw_only=True)
class Config(ScriptConfig):
    model: HookedModel
    train_config: TrainConfig = field(default_factory=TrainConfig)
    train_data: Dataset
    num_classes: int
    val_data: dict[str, Dataset] = field(default_factory=dict)
    # If True, returns the Lighting Trainer object (which has the model and a bunch
    # of other information, this may be useful when using interactively).
    # Otherwise (default), return only a dictionary of latest metrics, to avoid e.g.
    # submitit trying to pickle the entire Trainer object.
    return_trainer: bool = False

    def __post_init__(self):
        super().__post_init__()

        # For datasets that are not necessarily deterministic based only on
        # arguments, this is where validation sets are set to follow train_data
        if isinstance(self.train_data, BackdoorDataset):
            for name, val_config in self.val_data.items():
                # WanetBackdoor
                if (
                    isinstance(self.train_data.backdoor, WanetBackdoor)
                    and isinstance(val_config, BackdoorDataset)
                    and isinstance(val_config.backdoor, WanetBackdoor)
                ):
                    str_factor = (
                        val_config.backdoor.warping_strength
                        / self.train_data.backdoor.warping_strength
                    )
                    val_config.backdoor.control_grid = (
                        str_factor * self.train_data.backdoor.control_grid
                    )


@dataclass
class DebugConfig(Config):
    train_config: DebugTrainConfig = field(default_factory=DebugTrainConfig)
