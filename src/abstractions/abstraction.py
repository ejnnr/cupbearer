# An abstraction of a fixed computational graph (like a neural network) will
# take in a sequence of activations (with a fixed length). It learns one linear
# map for each activation, mapping it to an abstract representation. Additionally,
# it learns a non-linear map from each abstract representation to the next one.
# This non-linear map should output a *distribution* over the next abstract
# representation. Finally, it learns a linear map from the last abstract
# representation to the output of the computation.
# All of this is encapsulated in a flax module.

from dataclasses import field
import functools
from typing import Callable, List, Sequence
import flax.linen as nn
import jax
from abstractions import data


# A single computational step
Step = Callable
# A full computation is (for now) just a list of steps
# Could also be a graph (or of course non-static computations) in the future
Computation = List[Step]


def mlp_steps(output_dim: int, hidden_dims: Sequence[int]) -> Computation:
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


def cnn_steps(
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


class Abstraction(nn.Module):
    computation: Computation
    abstraction_maps: List[Step]

    @nn.compact
    def __call__(self, activations: List[jax.Array], train: bool = True):
        abstractions = [
            abstraction_map(x)
            for abstraction_map, x in zip(self.abstraction_maps, activations)
        ]

        input_map, *main_maps, output_map = self.computation
        *main_abstractions, output = abstractions
        predicted_abstractions = [
            step(x) for step, x in zip(main_maps, main_abstractions)
        ]

        predicted_output = output_map(output)

        return abstractions, predicted_abstractions, predicted_output


class MLPAbstraction(nn.Module):
    """An abstraction of a fixed computational graph (like a neural network)."""

    output_dim: int
    hidden_dims: Sequence[int] = field(default_factory=lambda: [256, 256])

    @nn.compact
    def __call__(self, activations: List[jax.Array], train: bool = True):
        # `train` is unused but our TrainerModule expects it to exist.

        abstractions = [
            nn.Dense(hidden_dim)(x)
            for x, hidden_dim in zip(activations, self.hidden_dims)
        ]
        # We skip the last abstract state, since there is no next one to predict
        predicted_abstractions = [
            nn.relu(nn.Dense(hidden_dim)(x))
            for x, hidden_dim in zip(abstractions[:-1], self.hidden_dims[1:])
        ]
        output = nn.Dense(self.output_dim)(abstractions[-1])

        return abstractions, predicted_abstractions, output


class CNNAbstraction(nn.Module):
    channels: Sequence[int] = field(default_factory=lambda: [32, 64])
    dense_dims: Sequence[int] = field(default_factory=lambda: [256])
    output_dim: int = 10

    @nn.compact
    def __call__(self, activations: List[jax.Array], train: bool = True):
        conv_activations = activations[: len(self.channels)]
        dense_activations = activations[len(self.channels) :]
        conv_abstractions = [
            nn.Conv(features=n_channels, kernel_size=(3, 3))(x)
            for x, n_channels in zip(conv_activations, self.channels)
        ]
        dense_abstractions = [
            nn.Dense(hidden_dim)(x)
            for x, hidden_dim in zip(dense_activations, self.dense_dims)
        ]
        predicted_abstractions = [
            nn.max_pool(
                nn.relu(nn.Conv(features=n_channels, kernel_size=(3, 3))(x)),
                window_shape=(2, 2),
                strides=(2, 2),
            )
            for x, n_channels in zip(conv_abstractions[:-1], self.channels[1:])
        ]
        # Special case for the last convolutional abstraction
        last_conv = conv_abstractions[-1]
        b, h, w, c = last_conv.shape
        last_conv = nn.max_pool(last_conv, window_shape=(h, w))
        last_conv = last_conv.squeeze(axis=(1, 2))
        assert last_conv.shape == (b, c)
        predicted_abstractions.append(nn.relu(nn.Dense(self.dense_dims[0])(last_conv)))
        predicted_abstractions += [
            nn.relu(nn.Dense(hidden_dim)(x))
            for x, hidden_dim in zip(dense_abstractions[:-1], self.dense_dims[1:])
        ]
        output = nn.Dense(self.output_dim)(dense_abstractions[-1])

        return conv_abstractions + dense_abstractions, predicted_abstractions, output


def abstraction_collate(model: nn.Module, params, return_original_batch=False):
    """Create a collate function that turns a dataloader into one for activations.

    You can use the output of this function as the collate_fn argument to any
    dataloader. Then the output of that dataloader won't be the data itself,
    but instead the outputs and activations of `model` on that data.

    Effictively, this isn't really a collate_fn, think of it more as a (pretty extreme)
    transform. The reason to implement this as a collate_fn is that we want to transform
    entire batches at once instead of running a forward pass of `model` on every
    single example.

    See https://github.com/pytorch/vision/issues/157#issuecomment-431289943 for the
    inspiration for this trick.

    WARNING: This function assumes that the dataset it is used for returns
    tuples/lists where the first element is the input to the model. (The other elements
    are ignored.)

    Args:
        model: The model to run on the data. It's call method must have a
            `return_activations` argument and return a tuple of (logits, activations)
            if `return_activations` is True.
        params: The parameters of the model to use.

    Returns:
        A function that takes a batch of data (which hasn't been collated yet)
        and returns a list of [logits, activations, original_batch], where logits is an
        array of shape (batch_size, *output_shape), activations is a list of arrays
        of shape (batch_size, *activation_shape), and original_batch is the collated
        version of the data itself (only if `return_original_batch` is True).
    """
    forward_fn = jax.jit(
        lambda x: model.apply({"params": params}, x, return_activations=True)
    )

    def mycollate(batch):
        collated = data.numpy_collate(batch)
        inputs = collated[0]
        logits, activations = forward_fn(inputs)
        if return_original_batch:
            return [logits, activations, collated]
        return [logits, activations]

    return mycollate
