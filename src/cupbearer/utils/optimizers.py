from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from cupbearer.utils.config_groups import register_config_group
from cupbearer.utils.utils import BaseConfig


@dataclass
class OptimizerConfig(BaseConfig, ABC):
    learning_rate: float = 1e-3

    @abstractmethod
    def build(self, params) -> torch.optim.Optimizer:
        pass


@dataclass
class Adam(OptimizerConfig):
    def build(self, params):
        return torch.optim.Adam(params, lr=self.learning_rate)


@dataclass
class SGD(OptimizerConfig):
    def build(self, params):
        return torch.optim.SGD(params, lr=self.learning_rate)


OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD,
}

register_config_group(OptimizerConfig, OPTIMIZERS)
