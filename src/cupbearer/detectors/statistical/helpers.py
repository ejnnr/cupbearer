from typing import Optional

import torch


def update_covariance(curr_mean, curr_C, curr_n, new_data):
    # Should be (batch, dim)
    assert new_data.ndim == 2

    new_n = len(new_data)
    total_n = curr_n + new_n

    new_mean = new_data.mean(dim=0)
    delta_mean = new_mean - curr_mean
    updated_mean = (curr_n * curr_mean + new_n * new_mean) / total_n

    delta_data = new_data - new_mean
    new_C = torch.einsum("bi,bj->ij", delta_data, delta_data)
    updated_C = (
        curr_C
        + new_C
        + curr_n * new_n / total_n * torch.einsum("i,j->ij", delta_mean, delta_mean)
    )

    return updated_mean, updated_C, total_n


def batch_covariance(batches):
    dim = batches[0].shape[1]
    mean = torch.zeros(dim)
    C = torch.zeros((dim, dim))
    n = 0

    for batch in batches:
        mean, C, n = update_covariance(mean, C, n, batch)

    return mean, C / (n - 1)  # Apply Bessel's correction for sample covariance


def mahalanobis(
    activations: dict[str, torch.Tensor],
    means: dict[str, torch.Tensor],
    inv_covariances: dict[str, torch.Tensor],
    inv_diag_covariances: Optional[dict[str, torch.Tensor]] = None,
):
    """Compute Simplified Relative Mahalanobis distances for a batch of activations.

    The Mahalanobis distance for each layer is computed,
    and the distances are then averaged over layers.

    Args:
        activations: Dictionary of activations for each layer,
            each element has shape (batch, dim)
        means: Dictionary of means for each layer, each element has shape (dim,)
        inv_covariances: Dictionary of inverse covariances for each layer,
            each element has shape (dim, dim)
        inv_diag_covariances: Dictionary of inverse diagonal covariances for each layer,
            each element has shape (dim,).
            If None, the usual Mahalanobis distance is computed instead of the
            (simplified) relative Mahalanobis distance.

    Returns:
        Dictionary of Mahalanobis distances for each layer,
        each element has shape (batch,).
    """
    distances: dict[str, torch.Tensor] = {}
    for k, activation in activations.items():
        batch_size = activation.shape[0]
        activation = activation.view(batch_size, -1)
        delta = activation - means[k]
        assert delta.ndim == 2 and delta.shape[0] == batch_size
        # Compute log likelihood under a Gaussian:
        distance = torch.einsum("bi,ij,bj->b", delta, inv_covariances[k], delta)
        if inv_diag_covariances is not None:
            distance -= torch.einsum("bi,i->b", delta**2, inv_diag_covariances[k])
        distances[k] = distance
    return distances


def quantum_entropy(
    whitened_activations: dict[str, torch.Tensor],
    alpha: float = 4,
) -> dict[str, torch.Tensor]:
    """Quantum ENtropy score per layer."""
    # TODO confusion:
    # In paper they whiten the matrix and then compute mean and covariance
    # matrix of whitened data, this means that either scores are not
    # independent between samples (score depends on other samples in batch) or
    # that mean is 0 and covariance matrix is identity matrix. Here I've
    # simplified it based on the second assumption.
    assert all(a.ndim == 2 for a in whitened_activations.values())
    return {
        k: whitened_activations[k].square().sum(dim=-1)
        / alpha ** (whitened_activations[k].size(-1))
        for k in whitened_activations
    }
