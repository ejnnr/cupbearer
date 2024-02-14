from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from cupbearer.data import Backdoor
from cupbearer.data.backdoor_data import BackdoorData
from cupbearer.models import StoredModel
from cupbearer.utils.scripts import load_config

from ._config import DebugTaskConfig, TaskConfig


@dataclass(kw_only=True)
class BackdoorDetection(TaskConfig):
    path: Path
    backdoor: Backdoor
    no_load: bool = False

    def _init_train_data(self):
        data_cfg = load_config(self.path, "train_data", BackdoorData)
        # Remove the backdoor
        self._train_data = data_cfg.original

    def _get_anomalous_test_data(self):
        copy = deepcopy(self._train_data)
        if not self.no_load:
            self.backdoor.load(self.path)
        return BackdoorData(original=copy, backdoor=self.backdoor)

    def _init_model(self):
        self._model = StoredModel(path=self.path)


@dataclass
class DebugBackdoorDetection(DebugTaskConfig, BackdoorDetection):
    pass
