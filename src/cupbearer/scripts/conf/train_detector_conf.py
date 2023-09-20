import os
from dataclasses import dataclass

from cupbearer.detectors import DetectorConfig
from cupbearer.tasks import TaskConfigBase
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.scripts import DirConfig, ScriptConfig
from simple_parsing.helpers import mutable_field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase = config_group(TaskConfigBase)
    detector: DetectorConfig = config_group(DetectorConfig)
    dir: DirConfig = mutable_field(
        DirConfig, base=os.path.join("logs", "train_detector")
    )
