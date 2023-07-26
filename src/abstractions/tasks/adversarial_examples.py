from dataclasses import dataclass

from torch.utils.data import Dataset
from abstractions.data.adversarial import AdversarialExampleConfig
from abstractions.data import TrainDataFromRun
from abstractions.models import StoredModel
from abstractions.models.computations import Model
from abstractions.utils.hydra import hydra_config
from . import TaskConfigBase


@hydra_config
@dataclass
class AdversarialExampleTask(TaskConfigBase):
    run_path: str

    def __post_init__(self):
        self._reference_data = TrainDataFromRun(path=self.run_path)
        self._anomalous_data = AdversarialExampleConfig(run_path=self.run_path)
        self._model = StoredModel(path=self.run_path)

    def get_anomalous_data(self) -> Dataset:
        return self._anomalous_data.get_dataset()

    def get_model(self) -> Model:
        return self._model.get_model()

    def get_reference_data(self) -> Dataset:
        return self._reference_data.get_dataset()
