import math
from dataclasses import dataclass
from pathlib import Path

from cupbearer.data import AdversarialExampleConfig, DatasetConfig, TrainDataFromRun
from cupbearer.models import ModelConfig, StoredModel

from ._config import DebugTaskConfig, TaskConfig


@dataclass
class AdversarialExampleTask(TaskConfig):
    path: Path
    attack_batch_size: int = 128
    success_threshold: float = 0.1
    steps: int = 40
    eps: float = 8 / 255

    def _get_clean_data(self, train: bool) -> DatasetConfig:
        if train:
            return TrainDataFromRun(path=self.path)
        else:
            return TrainDataFromRun(path=self.path).get_test_split()

    def _get_anomalous_data(self, train: bool) -> DatasetConfig:
        max_size = None
        if self.max_test_size:
            # This isn't strictly necessary, but it lets us avoid generating more
            # adversarial examples than needed.
            max_size = math.ceil(self.max_test_size * (1 - self.clean_test_weight))
        return AdversarialExampleConfig(
            path=self.path,
            max_size=max_size,
            attack_batch_size=self.attack_batch_size,
            success_threshold=self.success_threshold,
            steps=self.steps,
            eps=self.eps,
            use_test_data=not train,
        )

    def _get_model(self) -> ModelConfig:
        return StoredModel(path=self.path)


@dataclass(kw_only=True)
class DebugAdversarialExampleTask(DebugTaskConfig, AdversarialExampleTask):
    attack_batch_size: int = 1
    success_threshold: float = 1.0
    steps: int = 1
