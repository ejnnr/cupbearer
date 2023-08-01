from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from cupbearer.data._shared import TrainDataFromRun
from cupbearer.data.adversarial import AdversarialExampleConfig
from cupbearer.models import StoredModel
from cupbearer.models.computations import Model

from . import TaskConfigBase


@dataclass
class AdversarialExampleTask(TaskConfigBase):
    run_path: Path
    max_size: Optional[int] = None
    attack_batch_size: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self._reference_data = None
        self._anomalous_data = None
        self._model = None

    def _set_debug(self):
        super()._set_debug()
        self.max_size = 2
        self.attack_batch_size = 2

    @property
    def reference_data(self):
        if not self._reference_data:
            self._reference_data = TrainDataFromRun(
                path=self.run_path, max_size=self.max_size
            )
        return self._reference_data

    @property
    def anomalous_data(self):
        if not self._anomalous_data:
            self._anomalous_data = AdversarialExampleConfig(
                run_path=self.run_path,
                max_size=self.max_size,
                attack_batch_size=self.attack_batch_size,
            )
        return self._anomalous_data

    @property
    def model(self):
        if not self._model:
            self._model = StoredModel(path=self.run_path)
        return self._model

    def get_anomalous_data(self) -> Dataset:
        return self.anomalous_data.get_dataset()

    def get_model(self) -> Model:
        return self.model.get_model()

    def get_params(self) -> Model:
        return self.model.get_params()

    def get_reference_data(self) -> Dataset:
        return self.reference_data.get_dataset()
