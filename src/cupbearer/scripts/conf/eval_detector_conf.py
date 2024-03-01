from dataclasses import dataclass

from cupbearer.detectors import AnomalyDetector
from cupbearer.tasks import Task
from cupbearer.utils.scripts import ScriptConfig


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: Task
    detector: AnomalyDetector
    pbar: bool = False
