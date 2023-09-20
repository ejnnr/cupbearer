from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from cupbearer.data import (
    CornerPixelBackdoor,
    NoiseBackdoor,
    Transform,
    WanetBackdoor,
)
from cupbearer.data.backdoor_data import BackdoorData
from cupbearer.models import StoredModel
from cupbearer.utils.scripts import load_config

from . import TaskConfig


@dataclass(kw_only=True)
class BackdoorDetection(TaskConfig):
    run_path: Path
    no_load: bool = simple_parsing.field(action="store_true")
    backdoor: Transform = simple_parsing.subgroups(
        {
            "corner": CornerPixelBackdoor,
            "noise": NoiseBackdoor,
            "wanet": WanetBackdoor,
        }
    )

    def _init_train_data(self):
        # TODO: would be nice to use test data instead during eval
        data_cfg = load_config(self.run_path, "train_data", BackdoorData)
        # Remove the backdoor
        self._train_data = data_cfg.original

    def _get_anomalous_test_data(self):
        copy = deepcopy(self._train_data)
        if not self.no_load:
            self.backdoor.load(self.run_path)
        return BackdoorData(original=copy, backdoor=self.backdoor)

    def _init_model(self):
        self._model = StoredModel(path=self.run_path)
