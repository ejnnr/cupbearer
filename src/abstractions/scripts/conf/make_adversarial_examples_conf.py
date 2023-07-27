from dataclasses import dataclass
from typing import Optional

from abstractions.utils.scripts import ScriptConfig


@dataclass
class Config(ScriptConfig):
    batch_size: int = 128
    eps: float = 8 / 255
    max_examples: Optional[int] = None
