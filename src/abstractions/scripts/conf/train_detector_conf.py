import os
from dataclasses import dataclass

from abstractions.detectors import DetectorConfig
from abstractions.tasks import TaskConfigBase
from abstractions.utils.config_groups import config_group
from abstractions.utils.scripts import DirConfig, ScriptConfig
from simple_parsing import field
from simple_parsing.helpers import mutable_field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase = config_group(TaskConfigBase)
    detector: DetectorConfig = config_group(DetectorConfig)
    dir: DirConfig = mutable_field(
        DirConfig, base=os.path.join("logs", "train_detector")
    )
    debug: bool = field(action="store_true")
    debug_with_logging: bool = field(action="store_true")

    def __post_init__(self):
        if self.debug:
            self.debug_with_logging = True
            # Disable all file output.
            self.dir.base = None

        if self.debug_with_logging:
            self.detector.max_batch_size = 2
            if hasattr(self.detector.train, "num_epochs"):
                self.detector.train.num_epochs = 1  # type: ignore
            if hasattr(self.detector.train, "batch_size"):
                self.detector.train.batch_size = 2  # type: ignore
            if hasattr(self.detector.train, "max_steps"):
                self.detector.train.max_steps = 1  # type: ignore
