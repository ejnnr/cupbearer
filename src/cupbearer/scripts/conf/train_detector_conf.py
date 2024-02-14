from dataclasses import dataclass

from cupbearer.detectors import DetectorConfig
from cupbearer.tasks import TaskConfigBase
from cupbearer.utils.scripts import ScriptConfig
from cupbearer.utils.train import TrainConfig


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase
    detector: DetectorConfig
    train: TrainConfig

    def __post_init__(self):
        # We usually don't pass down paths, but in this case we ~always want these
        # to be the same and it would be easy to forget setting self.train.path
        # and then just not getting some logs.
        if self.path is not None and self.train.path is None:
            self.train.path = self.path
