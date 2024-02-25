from dataclasses import dataclass

from cupbearer.detectors import DetectorConfig, StoredDetector
from cupbearer.tasks import TaskConfigBase
from cupbearer.utils.scripts import ScriptConfig


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase
    detector: DetectorConfig | None = None
    save_config: bool = False
    pbar: bool = False

    def __post_init__(self):
        if self.detector is None:
            if self.path is None:
                raise ValueError("Path or detector must be set")
            self.detector = StoredDetector(path=self.path)
