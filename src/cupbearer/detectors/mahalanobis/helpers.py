from typing import Optional

import jax
import jax.numpy as jnp


def update_covariance(curr_mean, curr_C, curr_n, new_data):
    # Should be (batch, dim)
    assert new_data.ndim == 2

    new_n = len(new_data)
    total_n = curr_n + new_n

    new_mean = jnp.mean(new_data, axis=0)
    delta_mean = new_mean - curr_mean
    updated_mean = (curr_n * curr_mean + new_n * new_mean) / total_n

    delta_data = new_data - new_mean
    new_C = jnp.dot(delta_data.T, delta_data)
    updated_C = (
        curr_C + new_C + curr_n * new_n * jnp.outer(delta_mean, delta_mean) / total_n
    )

    return updated_mean, updated_C, total_n


def batch_covariance(batches):
    mean = jnp.zeros(batches[0].shape[1])
    C = jnp.zeros((batches[0].shape[1], batches[0].shape[1]))
    n = 0

    for batch in batches:
        mean, C, n = update_covariance(mean, C, n, batch)

    return mean, C / (n - 1)  # Apply Bessel's correction for sample covariance


def mahalanobis(
    activations: list[jax.Array],
    means: list[jax.Array],
    inv_covariances: list[jax.Array],
    inv_diag_covariances: Optional[list[jax.Array]] = None,
):
    """Compute Simplified Relative Mahalanobis distances for a batch of activations.

    The Mahalanobis distance for each layer is computed,
    and the distances are then averaged over layers.

    Args:
        activations: List of activations for each layer,
            each element has shape (batch, dim)
        means: List of means for each layer, each element has shape (dim,)
        inv_covariances: List of inverse covariances for each layer,
            each element has shape (dim, dim)
        inv_diag_covariances: List of inverse diagonal covariances for each layer,
            each element has shape (dim,).
            If None, the usual Mahalanobis distance is computed instead of the
            (simplified) relative Mahalanobis distance.

    Returns:
        Mahalanobis distance for each element in the batch, shape (batch,)
    """
    batch_size = activations[0].shape[0]
    distances: list[jax.Array] = []
    for i, activation in enumerate(activations):
        activation = activation.reshape(batch_size, -1)
        delta = activation - means[i]
        assert delta.ndim == 2 and delta.shape[0] == batch_size
        distance = jnp.sum((delta @ inv_covariances[i]) * delta, axis=1)
        if inv_diag_covariances is not None:
            distance -= jnp.sum(delta**2 * inv_diag_covariances[i][None], axis=1)
        distances.append(distance)
    return jnp.array(distances)
