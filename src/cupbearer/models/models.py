import math

from torch import nn

from .hooked_model import HookedModel


class MLP(HookedModel):
    def __init__(
        self,
        input_shape: list[int] | tuple[int, ...],
        output_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        in_features = math.prod(input_shape)
        self.layers = nn.ModuleDict()
        for i_layer, (in_features, out_features) in enumerate(
            zip([in_features] + hidden_dims, hidden_dims + [output_dim])
        ):
            self.layers[f"linear_{i_layer}"] = nn.Linear(
                in_features=in_features,
                out_features=out_features,
            )
            self.layers[f"relu_{i_layer}"] = nn.ReLU()
        self.layers.pop(next(reversed(self.layers.keys())))  # rm last relu

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for name, layer in self.layers.items():
            x = layer(x)
            self.store(f"post_{name}", x)
        return x

    @property
    def default_names(self) -> list[str]:
        return [
            f"post_{name}" for name in self.layers.keys() if name.startswith("linear")
        ]


class CNN(HookedModel):
    def __init__(
        self,
        input_shape: list[int] | tuple[int, ...],
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
        self.conv_layers = nn.ModuleDict()

        for i_layer, (in_channels, out_channels, kernel_size, stride) in enumerate(
            zip(
                [input_shape[0]] + channels[:-1],
                channels,
                self.kernel_sizes,
                self.strides,
            )
        ):
            self.conv_layers[f"conv_{i_layer}"] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
            )
            self.conv_layers[f"relu_{i_layer}"] = nn.ReLU()
            self.conv_layers[f"pool_{i_layer}"] = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.mlp = MLP((self.channels[-1],), self.output_dim, self.dense_dims)

    def forward(self, x):
        for name, layer in self.conv_layers.items():
            x = layer(x)
            self.store(f"conv_post_{name}", x)
        x = self.global_pool(x)
        self.store("post_global_pool", x)
        x = self.call_submodule("mlp", x)
        return x

    @property
    def default_names(self) -> list[str]:
        return [
            f"conv_post_{name}"
            for name in self.conv_layers.keys()
            if name.startswith("conv")
        ] + ["mlp_" + name for name in self.mlp.default_names]
