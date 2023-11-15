from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from cupbearer.utils.config_groups import register_config_group
from cupbearer.utils.scripts import load_config
from cupbearer.utils.utils import BaseConfig, PathConfigMixin, mutable_field

from .hooked_model import HookedModel
from .models import CNN, MLP


@dataclass(kw_only=True)
class ModelConfig(BaseConfig, ABC):
    @abstractmethod
    def build_model(self, input_shape: list[int] | tuple[int]) -> HookedModel:
        pass


@dataclass
class StoredModel(ModelConfig, PathConfigMixin):
    def build_model(self, input_shape) -> HookedModel:
        model_cfg = load_config(self.get_path(), "model", ModelConfig)
        model = model_cfg.build_model(input_shape)

        # Our convention is that LightningModules store the actual pytorch model
        # as a `model` attribute. We use the last checkpoint (generated via the
        # save_last=True option to the ModelCheckpoint callback).
        state_dict = torch.load(self.get_path() / "last.ckpt")["state_dict"]
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

    def build_model(self, input_shape: list[int] | tuple[int]) -> HookedModel:
        return MLP(
            input_shape=input_shape,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
        )

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            # TODO: we need at least two layers here because abstractions currently
            # only work in that case. Abstraction implementation should be fixed.
            self.hidden_dims = [2, 2]


@dataclass
class CNNConfig(ModelConfig):
    output_dim: int = 10
    channels: list[int] = mutable_field([32, 64])
    dense_dims: list[int] = mutable_field([256, 256])

    def build_model(self, input_shape: list[int] | tuple[int]) -> HookedModel:
        return CNN(
            input_shape=input_shape,
            output_dim=self.output_dim,
            channels=self.channels,
            dense_dims=self.dense_dims,
        )

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.channels = [2]
            self.dense_dims = [2]


MODELS = {
    "mlp": MLPConfig,
    "cnn": CNNConfig,
    "from_run": StoredModel,
}

register_config_group(ModelConfig, MODELS)
