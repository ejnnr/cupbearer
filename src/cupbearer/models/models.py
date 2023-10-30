import math

import torch.nn.functional as F
from torch import nn

from .hooked_model import HookedModel


class MLP(HookedModel):
    def __init__(
        self,
        input_shape: list[int] | tuple[int],
        output_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        in_features = math.prod(input_shape)
        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                )
                for in_features, out_features in zip(
                    [in_features] + hidden_dims, hidden_dims + [output_dim]
                )
            ]
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            self.store(f"post_linear.{i}", x)
            x = F.relu(x)
            self.store(f"post_relu.{i}", x)
        x = self.layers[-1](x)
        self.store(f"post_linear.{len(self.layers) - 1}", x)
        return x


class CNN(HookedModel):
    def __init__(
        self,
        input_shape: list[int] | tuple[int],
        output_dim: int,
        channels: list[int],
        dense_dims: list[int],
        kernel_sizes: list[int] | None = None,
        strides: list[int] | None = None,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.channels = channels
        self.dense_dims = dense_dims
        self.kernel_sizes = kernel_sizes if kernel_sizes else [3] * len(channels)
        self.strides = strides if strides else [1] * len(channels)
        self.conv_layers = nn.ModuleList()

        for in_channels, out_channels, kernel_size, stride in zip(
            [input_shape[0]] + channels[:-1],
            channels,
            self.kernel_sizes,
            self.strides,
        ):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding="same",
                )
            )

        self.mlp = MLP((self.channels[-1],), self.output_dim, self.dense_dims)

    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            self.store(f"conv.post_conv.{i}", x)
            x = F.relu(x)
            self.store(f"conv.post_relu.{i}", x)
            x = F.max_pool2d(x, 2)
            self.store(f"conv.post_pool.{i}", x)
        x = F.max_pool2d(x, x.shape[-2:])
        self.store("post_global_pool", x)
        x = self.call_submodule("mlp", x)
        return x
