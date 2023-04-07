import functools
from typing import Any, Mapping, Sequence
import flax.linen as nn

from abstractions.abstraction import Computation, Step


def linear(output_dim: int) -> Step:
    """A single linear layer."""
    return nn.Dense(features=output_dim)


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
            return [linear(output_dim=dim) for dim in hidden_dims]
        
        case {
            "_target_": "abstractions.computations.cnn",
            "output_dim": _,
            "channels": channels,
            "dense_dims": dense_dims,
        }:
            return [
                conv(output_dim=dim) for dim in channels
            ] + [linear(output_dim=dim) for dim in dense_dims]
        
        case _:
            raise ValueError(f"Unknown abstraction maps: {cfg}")


def mlp(output_dim: int, hidden_dims: Sequence[int]) -> Computation:
    """A simple feed-forward MLP."""
    steps = [
        # special case for first layer, since we need to flatten the input
        lambda x: nn.relu(
            nn.Dense(features=hidden_dims[0])(x.reshape((x.shape[0], -1)))
        ),
        # remaining hidden layers
        *[
            lambda x: nn.relu(nn.Dense(features=hidden_dim)(x))
            for hidden_dim in hidden_dims[1:]
        ],
        # output layer
        lambda x: nn.Dense(features=output_dim)(x),
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
