import copy
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


FONTS = ["Monaco", "DejaVu Sans Mono"]


class Step(ABC):
    output_dim: int = 0
    name: str = ""

    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        pass

    def get_drawable(self, orientation=Orientation.HORIZONTAL):
        try:
            from iceberg import Bounds, Colors, FontStyle
            from iceberg.primitives import Arrange, Rectangle, SimpleText
        except ImportError:
            raise ImportError(
                "You need to install iceberg-dsl to use the network visualization"
            )

        box = Rectangle(Bounds(size=(100, 100)), border_color=Colors.BLACK)
        text = None
        for font in FONTS:
            try:
                text = SimpleText(
                    text=self.name,
                    font_style=FontStyle(
                        family=font,
                        size=16,
                        color=Colors.BLACK,
                    ),
                )
                break
            except ValueError as e:
                if "Invalid font family" not in str(e):
                    raise
        if text is None:
            raise ValueError(f"Couldn't find a valid font, tried {FONTS}")

        info = self._info()
        if info is not None:
            info_text = SimpleText(
                text=info,
                font_style=FontStyle(
                    # This will still be the valid font we found earlier
                    family=font,  # type: ignore
                    size=16,
                    color=Colors.BLACK,
                ),
            )
            text = Arrange((text, info_text), Arrange.Direction.VERTICAL, gap=5)
        return box.add_centered(text)

    def _info(self):
        return None


# A full computation is (for now) just a list of steps
# Could also be a graph (or of course non-static computations) in the future
Computation = List[Step]


class Model(nn.Module):
    computation: Computation

    @nn.compact
    def __call__(self, x, return_activations=False, train=True):
        activations = []
        *main_maps, output_map = self.computation
        for step in main_maps:
            x = step(x)
            activations.append(x)

        x = output_map(x)

        if return_activations:
            return x, activations
        return x

    def get_drawable(self, return_nodes=False, layer_scores=None, inputs=None):
        return draw_computation(self.computation, return_nodes, layer_scores, inputs)


@dataclass
class Linear(Step):
    """A single linear layer."""

    output_dim: int
    name: str = "Linear"
    kernel_init: Any = nn.linear.default_kernel_init

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(features=self.output_dim, kernel_init=self.kernel_init)(x)

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
    kernel_init: Any = nn.linear.default_kernel_init

    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Conv(
            features=self.output_dim, kernel_size=(3, 3), kernel_init=self.kernel_init
        )(x)

    def _info(self):
        return f"d={self.output_dim}"


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

    for n_channels in channels:
        steps.append(ConvBlock(output_dim=n_channels))

    for i, hidden_dim in enumerate(dense_dims):
        steps.append(DenseBlock(output_dim=hidden_dim, is_first=(i == 0)))

    # output layer
    steps.append(Linear(output_dim=output_dim))
    return steps


@dataclass
class SoftmaxDrop(Step, nn.Module):
    """Scale each component of the input by a softmax of learned scores."""

    # output_dim=0 means derived from input
    output_dim: int = 0
    name: str = "Drop"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # TODO: The 3.0 should be configurable, and also maybe there's a more principled
        # thing to do, like just using zero? Idea here is to initialize close to keeping
        # everything, but not so close that gradients are tiny.
        scores = self.param("scores", constant_init, x.shape[1:], jnp.float_, 3.0)
        scores = nn.sigmoid(scores)
        # scores = nn.softmax(scores, axis=tuple(range(1, x.ndim + 1)))
        return x * scores[None]


def constant_init(
    key: jax.random.KeyArray, shape: Sequence[int], dtype=jnp.float_, value=0.0
) -> jax.Array:
    return jnp.full(shape, value, dtype=dtype)


def identity_init(
    key: jax.random.KeyArray, shape: Sequence[int], dtype=jnp.float_
) -> jax.Array:
    assert len(shape) >= 2
    assert shape[-1] == shape[-2]
    eye = jnp.eye(shape[-1], dtype=dtype)
    return jnp.broadcast_to(eye, shape)


def reduce_size_step(step: Step, factor: int) -> Step:
    new_step = copy.deepcopy(step)
    try:
        new_step.output_dim = round(step.output_dim / factor)
        if new_step.output_dim == 0:
            new_step.output_dim = 1
    except AttributeError:
        raise ValueError(f"Don't know how to reduce size of step {step}")

    return new_step


def reduce_size(
    computation: Computation, factor: int, output_dim: Optional[int] = None
) -> Computation:
    res = list(
        map(functools.partial(reduce_size_step, factor=factor), computation[:-1])
    )
    res.append(copy.deepcopy(computation[-1]))
    if output_dim is not None:
        try:
            res[-1].output_dim = output_dim
        except AttributeError:
            raise ValueError(f"Don't know how to reduce size of step {res[-1]}")

    return res


def make_image_grid(inputs):
    try:
        from iceberg.primitives import Grid, Image
    except ImportError:
        raise ImportError(
            "You need to install iceberg-dsl to use the network visualization"
        )
    # inputs is an array of shape (n_images, *image_dims).
    # We'll plot these inputs in a 3x3 grid.

    # We'll display at most 9 images.
    inputs = inputs[:9]

    if inputs[0].shape[-1] == 1:
        # grayscale images, copy channels
        inputs = np.repeat(inputs, 3, axis=-1)

    # Add alpha channel
    inputs = np.concatenate([inputs, np.ones_like(inputs[..., :1])], axis=-1)

    assert np.all(inputs >= 0) and np.all(inputs <= 1)
    inputs = (inputs * 255).astype(np.uint8)

    images = [Image(image=image) for image in inputs]
    # Restructure into list of lists:
    images = [images[i : i + 3] for i in range(0, len(images), 3)]
    GRID_SIZE = 200
    grid = Grid(images, gap=5)
    return grid.scale(GRID_SIZE / grid.bounds.width)


def draw_computation(
    computation: Computation, return_nodes=False, layer_scores=None, inputs=None
):
    try:
        from iceberg import (
            Bounds,
            Color,
            Colors,
            Corner,
            PathStyle,
        )
        from iceberg.arrows import Arrow
        from iceberg.primitives import (
            Arrange,
            Compose,
            Directions,
            Ellipse,
            Line,
        )
    except ImportError:
        raise ImportError(
            "You need to install iceberg-dsl to use the network visualization"
        )
    steps = [step.get_drawable() for step in computation]
    nodes = [
        Ellipse(Bounds(size=(50, 50)), border_color=Colors.BLACK) for _ in computation
    ]
    if layer_scores is not None:
        # The first node doesn't incur any loss in the current implementation,
        # so we don't color it.
        nodes[0].fill_color = Color(0.3, 0.3, 0.3)

        # TODO: might not make sense for all loss functions
        min_score = 0
        max_score = max(layer_scores)
        if max_score == min_score:
            normalized_scores = layer_scores
        else:
            normalized_scores = [
                (score - min_score) / (max_score - min_score) for score in layer_scores
            ]

        assert len(nodes) - 1 == len(normalized_scores), (
            f"len(nodes) - 1 = {len(nodes) - 1}"
            f"!= len(normalized_scores) = {len(normalized_scores)}"
        )
        for node, score in zip(nodes[1:], normalized_scores):
            node.fill_color = Color(1, 0, 0, score)

    # interleave the two lists:
    drawables = [x for pair in zip(steps, nodes) for x in pair]
    arranged = Arrange(drawables, gap=100)

    lines = []
    linestyle = PathStyle(color=Colors.BLACK)

    for a, b in zip(drawables[:-1], drawables[1:]):
        start = arranged.child_bounds(a).corners[Corner.MIDDLE_RIGHT]
        end = arranged.child_bounds(b).corners[Corner.MIDDLE_LEFT]
        if isinstance(a, Ellipse):
            line = Line(start, end, linestyle)
        else:
            line = Arrow(start, end, linestyle)
        lines.append(line)

    res = Compose((*lines, arranged))

    if inputs is not None:
        grid = make_image_grid(inputs)
        res = res.next_to(grid, Directions.LEFT * 20)

    if return_nodes:
        return res, nodes
    return res
