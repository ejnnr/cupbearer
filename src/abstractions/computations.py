from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Mapping, Sequence

import flax.linen as nn
import jax
from iceberg import Bounds, Colors, Drawable, FontStyle
from iceberg.primitives import Arrange, Rectangle, SimpleText


class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Step(ABC):
    name: str = ""

    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        pass

    def get_drawable(self, orientation=Orientation.HORIZONTAL) -> Drawable:
        box = Rectangle(Bounds(size=(100, 100)), border_color=Colors.BLACK)
        text = SimpleText(
            text=self.name,
            font_style=FontStyle(
                family="DejaVu Sans Mono",
                size=16,
                color=Colors.BLACK,
            ),
        )
        info = self._info()
        if info is not None:
            info_text = SimpleText(
                text=info,
                font_style=FontStyle(
                    family="DejaVu Sans Mono",
                    size=16,
                    color=Colors.BLACK,
                ),
            )
            text = Arrange((text, info_text), Arrange.Direction.VERTICAL, gap=5)
        return box.add_centered(text)

    def _info(self):
        return None


@dataclass
class Linear(Step):
    """A single linear layer."""

    output_dim: int
    name: str = "Linear"

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(features=self.output_dim)(x)

    def _info(self):
        return f"d={self.output_dim}"


@dataclass
class ReluLinear(Step):
    """A ReLU followed by a linear layer."""

    output_dim: int
    flatten_input: bool = False
    name: str = "Lin+ReLU"

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.flatten_input:
            x = x.reshape((x.shape[0], -1))
        return nn.relu(nn.Dense(features=self.output_dim)(x))

    def _info(self):
        return f"d={self.output_dim}"


@dataclass
class Conv(Step):
    """A single convolutional layer."""

    output_dim: int
    name: str = "Conv"

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Conv(features=self.output_dim, kernel_size=(3, 3))(x)

    def _info(self):
        return f"d={self.output_dim}"


def get_abstraction_maps(cfg: Mapping[str, Any]) -> list[Step]:
    """Get a list of abstraction maps from a model config."""
    match cfg:
        case {
            "_target_": "abstractions.computations.mlp",
            "output_dim": _,
            "hidden_dims": hidden_dims,
        }:
            return [Linear(output_dim=dim) for dim in hidden_dims]

        case {
            "_target_": "abstractions.computations.cnn",
            "output_dim": _,
            "channels": channels,
            "dense_dims": dense_dims,
        }:
            return [Conv(output_dim=dim) for dim in channels] + [
                Linear(output_dim=dim) for dim in dense_dims
            ]  # type: ignore

        case _:
            raise ValueError(f"Unknown abstraction maps: {cfg}")


# A full computation is (for now) just a list of steps
# Could also be a graph (or of course non-static computations) in the future
Computation = List[Step]


def mlp(output_dim: int, hidden_dims: Sequence[int]) -> Computation:
    """A simple feed-forward MLP."""
    steps = [
        # special case for first layer, since we need to flatten the input
        ReluLinear(output_dim=hidden_dims[0], flatten_input=True),
        # remaining hidden layers
        *[ReluLinear(output_dim=hidden_dim) for hidden_dim in hidden_dims[1:]],
        # output layer
        Linear(output_dim=output_dim),
    ]
    return steps


@dataclass
class ConvBlock(Step):
    """A single convolutional block."""

    output_dim: int
    name: str = "ConvBlock"

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Conv(features=self.output_dim, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x

    def _info(self):
        return f"d={self.output_dim}"


@dataclass
class DenseBlock(Step):
    """A single dense block."""

    output_dim: int
    is_first: bool = False
    name: str = "DenseBlock"

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.is_first:
            b, h, w, c = x.shape
            x = nn.max_pool(x, window_shape=(h, w))
            x = x.squeeze(axis=(1, 2))
            assert x.shape == (b, c)
        x = nn.relu(nn.Dense(features=self.output_dim)(x))
        return x

    def _info(self):
        return f"d={self.output_dim}"


def cnn(
    output_dim: int, channels: Sequence[int], dense_dims: Sequence[int]
) -> Computation:
    """A simple CNN."""
    steps = []

    for n_channels in channels:
        steps.append(ConvBlock(output_dim=n_channels))

    for i, hidden_dim in enumerate(dense_dims):
        steps.append(DenseBlock(output_dim=hidden_dim, is_first=(i == 0)))

    # output layer
    steps.append(Linear(output_dim=output_dim))
    return steps
