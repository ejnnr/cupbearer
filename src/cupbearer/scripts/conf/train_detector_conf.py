from dataclasses import dataclass

from cupbearer.detectors import AnomalyDetector
from cupbearer.tasks import Task
from cupbearer.utils.scripts import ScriptConfig
from cupbearer.utils.train import TrainConfig
from cupbearer.utils.utils import BaseConfig, mutable_field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: Task
    detector: AnomalyDetector
    num_classes: int
    train: BaseConfig = mutable_field(TrainConfig())
