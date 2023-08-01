from abc import ABC, abstractmethod
from dataclasses import dataclass

import optax
from cupbearer.utils.config_groups import register_config_group
from cupbearer.utils.utils import BaseConfig


@dataclass
class OptimizerConfig(BaseConfig, ABC):
    learning_rate: float = 1e-3

    @abstractmethod
    def build() -> optax.GradientTransformation:
        pass


@dataclass
class Adam(OptimizerConfig):
    def build(self) -> optax.GradientTransformation:
        return optax.adam(self.learning_rate)


OPTIMIZERS = {
    "adam": Adam,
}

register_config_group(OptimizerConfig, OPTIMIZERS)
