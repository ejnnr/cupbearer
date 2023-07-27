from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from abstractions.utils.config_groups import register_config_group
from abstractions.utils.scripts import load_config
from abstractions.utils.utils import BaseConfig, load, mutable_field

from .computations import Model, cnn, mlp


@dataclass
class ModelConfig(BaseConfig, ABC):
    @abstractmethod
    def get_model(self) -> Model:
        pass

    def get_params(self):
        return None


@dataclass
class StoredModel(ModelConfig):
    path: Path

    def get_model(self) -> Model:
        model_cfg = load_config(self.path, "model", ModelConfig)

        return model_cfg.get_model()

    def get_params(self):
        return load(self.path / "model")["params"]


@dataclass
class MLP(ModelConfig):
    output_dim: int = 10
    hidden_dims: list[int] = mutable_field([256, 256])

    def get_model(self) -> Model:
        return Model(mlp(output_dim=self.output_dim, hidden_dims=self.hidden_dims))


@dataclass
class CNN(ModelConfig):
    output_dim: int = 10
    channels: list[int] = mutable_field([32, 64])
    dense_dims: list[int] = mutable_field([256, 256])

    def get_model(self) -> Model:
        return Model(
            cnn(
                output_dim=self.output_dim,
                channels=self.channels,
                dense_dims=self.dense_dims,
            )
        )


MODELS = {
    "mlp": MLP,
    "cnn": CNN,
}

register_config_group(ModelConfig, MODELS)
