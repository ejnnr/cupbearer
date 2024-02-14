import math
from dataclasses import dataclass
from pathlib import Path

from cupbearer.data._shared import TrainDataFromRun
from cupbearer.data.adversarial import AdversarialExampleConfig
from cupbearer.models import StoredModel

from . import TaskConfig


@dataclass
class AdversarialExampleTask(TaskConfig):
    path: Path
    attack_batch_size: int = 128
    success_threshold: float = 0.1
    steps: int = 40
    eps: float = 8 / 255

    def _init_train_data(self):
        self._train_data = TrainDataFromRun(path=self.path)

    def _get_anomalous_test_data(self):
        max_size = None
        if self.max_test_size:
            # This isn't strictly necessary, but it lets us avoid generating more
            # adversarial examples than needed.
            max_size = math.ceil(self.max_test_size * (1 - self.normal_weight))
        return AdversarialExampleConfig(
            path=self.path,
            max_size=max_size,
            attack_batch_size=self.attack_batch_size,
            success_threshold=self.success_threshold,
            steps=self.steps,
            eps=self.eps,
        )

    def _init_model(self):
        self._model = StoredModel(path=self.path)


@dataclass(kw_only=True)
class DebugAdversarialExampleTask(AdversarialExampleTask):
    attack_batch_size: int = 1
    success_threshold: float = 1.0
    steps: int = 1
