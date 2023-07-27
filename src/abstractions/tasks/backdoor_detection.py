from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from torch.utils.data import Dataset

from abstractions.data import (
    CornerPixelBackdoor,
    NoiseBackdoor,
    TrainDataFromRun,
    Transform,
)
from abstractions.data.backdoor import BackdoorData
from abstractions.models import StoredModel
from abstractions.models.computations import Model

from . import TaskConfigBase


@dataclass
class BackdoorDetection(TaskConfigBase):
    run_path: Path
    backdoor: Transform = simple_parsing.subgroups(
        {
            "corner": CornerPixelBackdoor,
            "noise": NoiseBackdoor,
        }
    )

    def __post_init__(self):
        # TODO: actually, we need to remove backdoors here
        # TODO: would be nice to use test data instead during eval
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

    def get_params(self) -> Model:
        return self._model.get_params()

    def get_reference_data(self) -> Dataset:
        return self._reference_data.get_dataset()
