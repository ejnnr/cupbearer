from dataclasses import dataclass

from cupbearer.detectors import DetectorConfig
from cupbearer.tasks import TaskConfig
from cupbearer.utils.scripts import ScriptConfig


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfig
    detector: DetectorConfig
