from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf
import hydra.utils

from abstractions.utils.hydra import hydra_config, hydra_config_base
from abstractions.utils.utils import mutable_field

from .computations import Model, mlp, cnn


@hydra_config_base("model")  # type: ignore
@dataclass
class ModelConfig(ABC):
    @abstractmethod
    def get_model(self) -> Model:
        pass


@hydra_config
@dataclass
class StoredModel(ModelConfig):
    path: str

    def get_model(self) -> Model:
        run = Path(self.path)
        cfg = OmegaConf.load(
            hydra.utils.to_absolute_path(str(run / ".hydra" / "config.yaml"))
        )
        cfg = OmegaConf.to_object(cfg)

        if not hasattr(cfg, "model"):
            raise ValueError(f"Expected model to be in config, got {cfg}")
        if not isinstance(cfg.model, ModelConfig):  # type: ignore
            raise ValueError(
                f"Expected model to be a ModelConfig, got {cfg.model}"  # type: ignore
            )

        return cfg.model.get_model()  # type: ignore


@hydra_config
@dataclass
class MLP(ModelConfig):
    output_dim: int = 10
    hidden_dims: list[int] = mutable_field([256, 256])

    def get_model(self) -> Model:
        return Model(mlp(output_dim=self.output_dim, hidden_dims=self.hidden_dims))
