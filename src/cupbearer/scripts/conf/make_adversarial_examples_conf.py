from dataclasses import dataclass
from typing import Optional

from cupbearer.utils.scripts import ScriptConfig


@dataclass
class Config(ScriptConfig):
    batch_size: int = 128
    eps: float = 8 / 255
    max_examples: Optional[int] = None
    success_threshold: float = 0.1
    save_config: bool = False

    def _set_debug(self):
        super()._set_debug()
        self.max_examples = 2
        self.batch_size = 2
        # Can't reliably expect the attack to succeed with toy debug settings:
        self.success_threshold = 1.0
