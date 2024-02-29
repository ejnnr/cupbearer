from dataclasses import dataclass
from pathlib import Path

from cupbearer.data import Backdoor, DatasetConfig
from cupbearer.data.backdoor_data import BackdoorData
from cupbearer.models import ModelConfig, StoredModel
from cupbearer.utils.scripts import load_config

from ._config import DebugTaskConfig, TaskConfig


@dataclass(kw_only=True)
class BackdoorDetection(TaskConfig):
    path: Path
    backdoor: Backdoor
    no_load: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.backdoored_train_data = load_config(self.path, "train_data", BackdoorData)

    def _get_clean_data(self, train: bool) -> DatasetConfig:
        if train:
            return self.backdoored_train_data.original
        else:
            return self.backdoored_train_data.original.get_test_split()

    def _get_anomalous_data(self, train: bool) -> DatasetConfig:
        if not self.no_load:
            self.backdoor.load(self.path)

        # TODO: should we get rid of `self.backdoor` and just use the existing one
        # from the training run?
        return BackdoorData(
            original=self._get_clean_data(train), backdoor=self.backdoor
        )

    def _get_model(self) -> ModelConfig:
        return StoredModel(path=self.path)


@dataclass
class DebugBackdoorDetection(DebugTaskConfig, BackdoorDetection):
    pass
