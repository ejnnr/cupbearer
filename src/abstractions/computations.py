from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Any, List, Mapping, Sequence
import flax.linen as nn
from enum import Enum
from iceberg import Drawable, Bounds, Colors, PathStyle, FontStyle
from iceberg.primitives import Rectangle, Line, Directions, Arrange, SimpleText
import jax


class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Step(ABC):
    name: str = ""

    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        pass

    def get_drawable(self, orientation=Orientation.HORIZONTAL) -> Drawable:
        step = self._get_drawable(orientation)
        style = PathStyle(Colors.BLACK)
        if orientation == Orientation.HORIZONTAL:
            in_arrow = Line((0, 0), (100, 0), style)
            out_arrow = Line((0, 0), (100, 0), style)
            return Arrange([in_arrow, step, out_arrow], gap=0)
        else:
            in_arrow = Line((0, 0), (0, 100), style)
            out_arrow = Line((0, 0), (0, 100), style)
            return Arrange(
                [in_arrow, step, out_arrow], Arrange.Direction.VERTICAL, gap=0
            )

    def _get_drawable(self, orientation: Orientation) -> Drawable:
        text = SimpleText(
            text=self.name,
            font_style=FontStyle(
                family="Arial",
                size=16,
                color=Colors.BLACK,
            ),
        )
        box = Rectangle(Bounds(size=(100, 100)), border_color=Colors.BLACK)
        return box.add_centered(text)


@dataclass
class Linear(Step):
    """A single linear layer."""

    output_dim: int
    name: str = "Linear"

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(features=self.output_dim)(x)


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


def conv(output_dim: int) -> Step:
    """A single convolutional layer."""
    return nn.Conv(features=output_dim, kernel_size=(3, 3))


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
            return [conv(output_dim=dim) for dim in channels] + [
                Linear(output_dim=dim) for dim in dense_dims
            ]

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


def cnn(
    output_dim: int, channels: Sequence[int], dense_dims: Sequence[int]
) -> Computation:
    """A simple CNN."""
    steps = []

    # Convolutional layers
    def conv_block(x, n_channels):
        x = nn.Conv(features=n_channels, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x

    for n_channels in channels:
        steps.append(functools.partial(conv_block, n_channels=n_channels))

    # Dense layers
    def dense_block(x, hidden_dim, is_first):
        if is_first:
            b, h, w, c = x.shape
            x = nn.max_pool(x, window_shape=(h, w))
            x = x.squeeze(axis=(1, 2))
            assert x.shape == (b, c)
        x = nn.relu(nn.Dense(features=hidden_dim)(x))
        return x

    for i, hidden_dim in enumerate(dense_dims):
        steps.append(
            functools.partial(dense_block, hidden_dim=hidden_dim, is_first=i == 0)
        )

    # output layer
    steps.append(lambda x: nn.Dense(features=output_dim)(x))
    return steps
