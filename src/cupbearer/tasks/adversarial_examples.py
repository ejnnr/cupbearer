import math
from dataclasses import dataclass
from typing import Optional

from cupbearer.data._shared import TrainDataFromRun
from cupbearer.data.adversarial import AdversarialExampleConfig
from cupbearer.models import StoredModel

from . import TaskConfig


@dataclass
class AdversarialExampleTask(TaskConfig):
    attack_batch_size: Optional[int] = None
    success_threshold: float = 0.1

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.attack_batch_size = 1
            # TODO: This and other configs are duplicated several times between here,
            # the AdversarialExampleConfig, the AdversarialExampleDataset, and the
            # make_adversarial_examples script. In need of refactoring.
            self.success_threshold = 1.0

    def _init_train_data(self):
        self._train_data = TrainDataFromRun(path=self.get_path())

    def _get_anomalous_test_data(self):
        max_size = None
        if self.max_test_size:
            # This isn't strictly necessary, but it lets us avoid generating more
            # adversarial examples than needed.
            max_size = math.ceil(self.max_test_size * (1 - self.normal_weight))
        return AdversarialExampleConfig(
            path=self.get_path(),
            max_size=max_size,
            attack_batch_size=self.attack_batch_size,
            success_threshold=self.success_threshold,
        )

    def _init_model(self):
        self._model = StoredModel(path=self.get_path())
