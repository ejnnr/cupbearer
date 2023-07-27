from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from abstractions.data._shared import TrainDataFromRun
from abstractions.data.adversarial import AdversarialExampleConfig
from abstractions.models import StoredModel
from abstractions.models.computations import Model

from . import TaskConfigBase


@dataclass
class AdversarialExampleTask(TaskConfigBase):
    run_path: Path

    def __post_init__(self):
        self._reference_data = TrainDataFromRun(path=self.run_path)
        self._anomalous_data = AdversarialExampleConfig(run_path=self.run_path)
        self._model = StoredModel(path=self.run_path)

    def get_anomalous_data(self) -> Dataset:
        return self._anomalous_data.get_dataset()

    def get_model(self) -> Model:
        return self._model.get_model()

    def get_params(self) -> Model:
        return self._model.get_params()

    def get_reference_data(self) -> Dataset:
        return self._reference_data.get_dataset()
