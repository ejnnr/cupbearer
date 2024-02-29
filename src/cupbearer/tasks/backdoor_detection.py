from dataclasses import dataclass
from pathlib import Path

from cupbearer.data import DatasetConfig
from cupbearer.data.backdoor_data import BackdoorData
from cupbearer.models import ModelConfig, StoredModel
from cupbearer.utils.scripts import load_config

from ._config import DebugTaskConfig, FuzzedTask


@dataclass(kw_only=True)
class BackdoorDetection(FuzzedTask):
    path: Path
    no_load: bool = False

    def __post_init__(self):
        backdoor_data = load_config(self.path, "train_data", BackdoorData)
        self._original = backdoor_data.original
        self._backdoor = backdoor_data.backdoor
        self._backdoor.p_backdoor = 1.0

        if not self.no_load:
            self._backdoor.load(self.path)

        # Call this only now that _original and _backdoor are set.
        super().__post_init__()

    def _get_base_data(self) -> DatasetConfig:
        return self._original

    def fuzz(self, data: DatasetConfig) -> DatasetConfig:
        return BackdoorData(original=data, backdoor=self._backdoor)

    def _get_model(self) -> ModelConfig:
        return StoredModel(path=self.path)


@dataclass
class DebugBackdoorDetection(DebugTaskConfig, BackdoorDetection):
    pass
