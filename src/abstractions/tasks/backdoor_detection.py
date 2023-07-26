from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from torch.utils.data import Dataset
from abstractions.data.backdoor import BackdoorData
from abstractions.data import TrainDataFromRun
from abstractions.models import StoredModel
from abstractions.models.computations import Model

from abstractions.tasks import TaskConfig
from abstractions.utils.hydra import hydra_config


@hydra_config
@dataclass
class BackdoorDetection(TaskConfig):
    run_path: str
    backdoor: dict[str, Any]

    def __post_init__(self):
        self._reference_data = TrainDataFromRun(path=self.run_path)
        self._anomalous_data = deepcopy(self._reference_data)
        self._anomalous_data = BackdoorData(
            original=self._anomalous_data, backdoor=self.backdoor
        )
        self._model = StoredModel(path=self.run_path)

    def get_anomalous_data(self) -> Dataset:
        return self._anomalous_data.get_dataset()

    def get_model(self) -> Model:
        return self._model.get_model()

    def get_reference_data(self) -> Dataset:
        return self._reference_data.get_dataset()
