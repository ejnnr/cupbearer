# An abstraction of a fixed computational graph (like a neural network) will
# take in a sequence of activations (with a fixed length). It learns one linear
# map for each activation, mapping it to an abstract representation. Additionally,
# it learns a non-linear map from each abstract representation to the next one.
# This non-linear map should output a *distribution* over the next abstract
# representation. Finally, it learns a linear map from the last abstract
# representation to the output of the computation.
# All of this is encapsulated in a flax module.

from dataclasses import field
from typing import List, Sequence
import flax.linen as nn
import jax
from abstractions import data


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
