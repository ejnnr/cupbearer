from dataclasses import dataclass
from typing import Optional

from cupbearer.detectors import DetectorConfig
from cupbearer.tasks import TaskConfigBase
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.scripts import ScriptConfig


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase = config_group(TaskConfigBase)
    detector: DetectorConfig = config_group(DetectorConfig)
    max_size: Optional[int] = None

    def _set_debug(self):
        super()._set_debug()
        self.max_size = 2
