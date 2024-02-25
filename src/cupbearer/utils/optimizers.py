from dataclasses import dataclass

import torch


@dataclass
class OptimizerConfigMixin:
    """Optimizer settings, meant to be used as a mix-in in other configs."""

    optimizer: str = "adam"
    lr: float = 1e-3

    def get_optimizer(self, params) -> torch.optim.Optimizer:
        if self.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.lr)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
