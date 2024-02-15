from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch

from cupbearer.data import DataFormat, TensorDataFormat, TextDataFormat
from cupbearer.utils.scripts import load_config
from cupbearer.utils.utils import BaseConfig, mutable_field

from .hooked_model import HookedModel
from .models import CNN, MLP
from .transformers import ClassifierTransformer


@dataclass(kw_only=True)
class ModelConfig(BaseConfig, ABC):
    @abstractmethod
    def build_model(self, input_format: DataFormat) -> HookedModel:
        pass


@dataclass
class StoredModel(ModelConfig):
    path: Path

    def build_model(self, input_format) -> HookedModel:
        model_cfg = load_config(self.path, "model", ModelConfig)
        model = model_cfg.build_model(input_format)

        # Our convention is that LightningModules store the actual pytorch model
        # as a `model` attribute. We use the last checkpoint (generated via the
        # save_last=True option to the ModelCheckpoint callback).
        state_dict = torch.load(self.path / "checkpoints" / "last.ckpt")["state_dict"]
        # We want the state_dict for the 'model' submodule, so remove
        # the 'model.' prefix from the keys.
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
        assert isinstance(model, torch.nn.Module)
        model.load_state_dict(state_dict)
        return model


@dataclass
class MLPConfig(ModelConfig):
    output_dim: int = 10
    hidden_dims: list[int] = mutable_field([256, 256])

    def build_model(self, input_format: DataFormat) -> HookedModel:
        if not isinstance(input_format, TensorDataFormat):
            raise ValueError(f"MLP only supports tensor input, got {input_format}")
        return MLP(
            input_shape=input_format.shape,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
        )


@dataclass
class DebugMLPConfig(MLPConfig):
    # TODO: we need at least two layers here because abstractions currently
    # only work in that case. Abstraction implementation should be fixed.
    # Additionally, we make network with some width to reduce chance that all
    # neurons are dead.
    hidden_dims: list[int] = mutable_field([5, 5])


@dataclass
class CNNConfig(ModelConfig):
    output_dim: int = 10
    channels: list[int] = mutable_field([32, 64])
    dense_dims: list[int] = mutable_field([256, 256])

    def build_model(self, input_format: DataFormat) -> HookedModel:
        if not isinstance(input_format, TensorDataFormat):
            raise ValueError(f"CNN only supports tensor input, got {input_format}")
        return CNN(
            input_shape=input_format.shape,
            output_dim=self.output_dim,
            channels=self.channels,
            dense_dims=self.dense_dims,
        )


@dataclass
class TransformerConfig(ModelConfig):
    model: str = "pythia-14m"
    num_classes: int = 2
    device: str | None = None

    def build_model(self, input_format: DataFormat):
        if not isinstance(input_format, TextDataFormat):
            raise ValueError(
                f"Transformers only support text input, got {type(input_format)}"
            )
        return ClassifierTransformer(self.model, self.num_classes, device=self.device)


@dataclass
class DebugCNNConfig(CNNConfig):
    channels: list[int] = mutable_field([2])
    dense_dims: list[int] = mutable_field([2])
