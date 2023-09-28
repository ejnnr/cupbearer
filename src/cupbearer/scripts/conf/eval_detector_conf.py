from dataclasses import dataclass

from cupbearer.detectors import DetectorConfig, StoredDetector
from cupbearer.tasks import TaskConfigBase
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.scripts import ScriptConfig
from simple_parsing import field


@dataclass(kw_only=True)
class Config(ScriptConfig):
    task: TaskConfigBase = config_group(TaskConfigBase)
    detector: DetectorConfig = config_group(
        DetectorConfig, default_factory=StoredDetector
    )
    save_config: bool = False
    pbar: bool = field(action="store_true")
