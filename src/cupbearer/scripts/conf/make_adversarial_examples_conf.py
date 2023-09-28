from dataclasses import dataclass
from typing import Optional

from cupbearer.data import TrainDataFromRun
from cupbearer.models import StoredModel
from cupbearer.utils.scripts import ScriptConfig


@dataclass
class Config(ScriptConfig):
    batch_size: int = 128
    eps: float = 8 / 255
    max_examples: Optional[int] = None
    success_threshold: float = 0.1
    save_config: bool = False

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.max_examples = 2
            self.batch_size = 2
            # Can't reliably expect the attack to succeed with toy debug settings:
            self.success_threshold = 1.0

        if self.dir.path is None:
            raise ValueError("Must specify a run path")

        self.model = StoredModel(path=self.dir.path)
        self.data = TrainDataFromRun(path=self.dir.path)
