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
    activation: torch.Tensor,
    mean: torch.Tensor,
    inv_covariance: torch.Tensor,
    inv_diag_covariance: Optional[torch.Tensor] = None,
):
    """Compute Simplified Relative Mahalanobis distances for a batch of activations.

    The Mahalanobis distance for each layer is computed,
    and the distances are then averaged over layers.

    Args:
        activation: values to compute distance for, with shape (batch, dim)
        mean: mean of shape (dim,)
        inv_covariance: inverse covariance matrix of shape (dim, dim)
        inv_diag_covariance: Tensor of shape (dim,).
            If None, the usual Mahalanobis distance is computed
            instead of the (simplified) relative Mahalanobis distance.

    Returns:
        Tensor of shape (batch,) with the Mahalanobis distances.
    """
    batch_size = activation.shape[0]
    activation = activation.view(batch_size, -1)
    delta = activation - mean
    assert delta.ndim == 2 and delta.shape[0] == batch_size
    # Compute unnormalized negative log likelihood under a Gaussian:
    distance = torch.einsum("bi,ij,bj->b", delta, inv_covariance, delta)
    if inv_diag_covariance is not None:
        distance -= torch.einsum("bi,i->b", delta**2, inv_diag_covariance)
    return distance


def quantum_entropy(
    whitened_activations: torch.Tensor,
    alpha: float = 4,
) -> torch.Tensor:
    """Quantum Entropy score.

    Args:
        whitened_activations: whitened activations, with shape (batch, dim)
        alpha: QUE hyperparameter
    """
    # Compute QUE-score
    centered_batch = whitened_activations - whitened_activations.mean(
        dim=0, keepdim=True
    )
    batch_cov = centered_batch.mT @ centered_batch

    batch_cov_norm = torch.linalg.eigvalsh(batch_cov).max()
    exp_factor = torch.matrix_exp(alpha * batch_cov / batch_cov_norm)

    return torch.einsum(
        "bi,ij,jb->b",
        whitened_activations,
        exp_factor,
        whitened_activations.mT,
    )
