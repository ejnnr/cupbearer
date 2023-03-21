# An abstraction of a fixed computational graph (like a neural network) will
# take in a sequence of activations (with a fixed length). It learns one linear
# map for each activation, mapping it to an abstract representation. Additionally,
# it learns a non-linear map from each abstract representation to the next one.
# This non-linear map should output a *distribution* over the next abstract
# representation. Finally, it learns a linear map from the last abstract
# representation to the output of the computation.
# All of this is encapsulated in a flax module.

from typing import List
import flax.linen as nn
import jax.numpy as jnp
import jax


class Abstraction(nn.Module):
    """An abstraction of a fixed computational graph (like a neural network)."""

    abstract_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, activations: List[jax.Array]):
        abstractions = [nn.Dense(self.abstract_dim)(x) for x in activations]
        # We skip the last abstract state, since there is no next one to predict
        predicted_abstractions = [
            nn.relu(nn.Dense(self.abstract_dim)(x)) for x in abstractions[:-1]
        ]
        output = nn.Dense(self.output_dim)(abstractions[-1])

        return abstractions, predicted_abstractions, output
