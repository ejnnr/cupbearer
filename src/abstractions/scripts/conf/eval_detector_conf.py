from dataclasses import dataclass
from typing import Optional

from abstractions.detectors import DetectorConfig
from abstractions.tasks import TaskConfigBase
from abstractions.utils.config_groups import config_group
from abstractions.utils.scripts import ScriptConfig
from simple_parsing import field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase = config_group(TaskConfigBase)
    detector: DetectorConfig = config_group(DetectorConfig)
    max_size: Optional[int] = None
    debug: bool = field(action="store_true")
    debug_with_logging: bool = field(action="store_true")

    def __post_init__(self):
        if self.debug:
            self.debug_with_logging = True
            # Disable all file output.
            self.dir.base = None

        if self.debug_with_logging:
            self.detector.max_batch_size = 2
            self.max_size = 2
