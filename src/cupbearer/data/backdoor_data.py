# This needs to be in a separate file from backdoors.py because of circularity issues
# with the config groups. See __init__.py.
from dataclasses import dataclass

from cupbearer.data import DatasetConfig
from cupbearer.data._shared import Transform
from cupbearer.utils.config_groups import config_group


@dataclass
class BackdoorData(DatasetConfig):
    original: DatasetConfig = config_group(DatasetConfig)
    backdoor: Transform = config_group(Transform)

    def __post_init__(self):
        self.transforms = {
            **self.original.transforms,
            **self.transforms,
            "backdoor": self.backdoor,
        }

    def _build(self):
        return self.original._build()
