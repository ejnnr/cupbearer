from dataclasses import dataclass

import torch

from cupbearer.utils.utils import BaseConfig


@dataclass
class OptimizerConfig(BaseConfig):
    name: str = "adam"
    lr: float = 1e-3

    def get_optimizer(self, params) -> torch.optim.Optimizer:
        if self.name == "adam":
            return torch.optim.Adam(params, lr=self.lr)
        elif self.name == "sgd":
            return torch.optim.SGD(params, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer {self.name}")
