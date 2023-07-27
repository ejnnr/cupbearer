import os
from dataclasses import dataclass

from abstractions.detectors import DetectorConfig
from abstractions.tasks import TaskConfigBase
from abstractions.utils.config_groups import config_group
from abstractions.utils.scripts import DirConfig, ScriptConfig
from simple_parsing.helpers import mutable_field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase = config_group(TaskConfigBase)
    detector: DetectorConfig = config_group(DetectorConfig)
    dir: DirConfig = mutable_field(
        DirConfig, base=os.path.join("logs", "train_detector")
    )
