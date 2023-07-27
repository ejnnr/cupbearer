from dataclasses import dataclass

from abstractions.detectors import DetectorConfig
from abstractions.tasks import TaskConfigBase
from abstractions.utils.config_groups import config_group
from abstractions.utils.scripts import ScriptConfig


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase = config_group(TaskConfigBase)
    detector: DetectorConfig = config_group(DetectorConfig)
