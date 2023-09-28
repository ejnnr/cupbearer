from dataclasses import dataclass

from cupbearer.data.toy_ambiguous_features import ToyFeaturesConfig
from cupbearer.models import StoredModel

from . import TaskConfig


@dataclass
class ToyFeaturesTask(TaskConfig):
    noise: float = 0.1

    def _init_train_data(self):
        self._train_data = ToyFeaturesConfig(correlated=True, noise=self.noise)

    def _get_anomalous_test_data(self):
        return ToyFeaturesConfig(correlated=False, noise=self.noise)

    def _init_model(self):
        self._model = StoredModel(path=self.get_path())
