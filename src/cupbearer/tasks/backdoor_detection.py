from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import simple_parsing
from torch.utils.data import Dataset

from cupbearer.data import (
    CornerPixelBackdoor,
    NoiseBackdoor,
    Transform,
)
from cupbearer.data.backdoor_data import BackdoorData
from cupbearer.models import StoredModel
from cupbearer.models.computations import Model
from cupbearer.utils.scripts import load_config

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
    max_size: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        # We'll only actually instantiate these when we need them, in case relevant
        # attributes get changed after initialization.
        self._reference_data = None
        self._anomalous_data = None
        self._model = None

    def _set_debug(self):
        super()._set_debug()
        self.max_size = 2

    @property
    def reference_data(self):
        # TODO: would be nice to use test data instead during eval
        if not self._reference_data:
            data_cfg = load_config(self.run_path, "train_data", BackdoorData)
            # Remove the backdoor
            self._reference_data = data_cfg.original
            self._reference_data.max_size = self.max_size
        return self._reference_data

    @property
    def anomalous_data(self):
        if not self._anomalous_data:
            copy = deepcopy(self.reference_data)
            self._anomalous_data = BackdoorData(original=copy, backdoor=self.backdoor)
        return self._anomalous_data

    @property
    def model(self):
        if not self._model:
            self._model = StoredModel(path=self.run_path)
        return self._model

    def build_anomalous_data(self) -> Dataset:
        return self.anomalous_data.build()

    def build_model(self) -> Model:
        return self.model.build_model()

    def build_params(self) -> Model:
        return self.model.build_params()

    def build_reference_data(self) -> Dataset:
        return self.reference_data.build()
