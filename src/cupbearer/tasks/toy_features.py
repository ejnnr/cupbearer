from dataclasses import dataclass
from pathlib import Path

from cupbearer.data.toy_ambiguous_features import ToyFeaturesConfig
from cupbearer.models import StoredModel

from ._config import DebugTaskConfig, TaskConfig


@dataclass
class ToyFeaturesTask(TaskConfig):
    path: Path
    noise: float = 0.1

    def _init_train_data(self):
        self._train_data = ToyFeaturesConfig(correlated=True, noise=self.noise)

    def _get_anomalous_test_data(self):
        return ToyFeaturesConfig(correlated=False, noise=self.noise)

    def _init_model(self):
        self._model = StoredModel(path=self.path)


@dataclass
class DebugToyFeaturesTask(DebugTaskConfig, ToyFeaturesTask):
    pass