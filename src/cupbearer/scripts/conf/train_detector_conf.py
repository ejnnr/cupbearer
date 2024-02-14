from dataclasses import dataclass, field

from cupbearer.detectors import DetectorConfig
from cupbearer.tasks import TaskConfigBase
from cupbearer.utils.scripts import ScriptConfig
from cupbearer.utils.train import DebugTrainConfig, TrainConfig


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase
    detector: DetectorConfig
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class DebugConfig(Config):
    train: TrainConfig = field(default_factory=DebugTrainConfig)
