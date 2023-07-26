from abc import ABC
from dataclasses import dataclass
from typing import Any
from hydra.conf import HydraConf
from abstractions.utils.hydra import hydra_config, hydra_config_base

from abstractions.utils.utils import mutable_field, dict_field

SCRIPT_DEFAULTS = [{"override hydra/launcher": "submitit_slurm"}]


@dataclass
class ScriptConfig(ABC):
    # hydra: HydraConf
    shards: int = 4

    dir: dict[str, Any] = mutable_field(
        {"log": "logs/${hydra.job.name}", "run": "${now:%Y-%m-%d_%H-%M-%S}"}
    )
    seed: int = 0


@hydra_config_base("optim")  # type: ignore
@dataclass
class OptimizerConfig(ABC):
    _target_: str
    learning_rate: float = 1e-3


@hydra_config
@dataclass
class Adam(OptimizerConfig):
    _target_: str = "optax.adam"
