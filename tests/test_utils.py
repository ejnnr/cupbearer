import math

import pytest
import torch
from cupbearer.detectors.statistical.helpers import batch_covariance


@pytest.mark.parametrize("N", [10, 15, 100])
def test_batch_covariance(N: int):
    # Create synthetic data with non-trivial covariance
    torch.manual_seed(42)
    mean = torch.tensor([1.0, -2.0, 3.0])
    cov = torch.tensor([[2.0, 1.0, 0.5], [1.0, 4.0, 0.0], [0.5, 0.0, 3.0]])
    dist = torch.distributions.MultivariateNormal(mean, cov)
    data = dist.sample((N,))

    # Split data into batches
    batch_size = 9
    # Round up, so we'll have a final batch with smaller size to also test that case
    num_batches = math.ceil(len(data) / batch_size)
    batches = [data[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]

    # Compute covariance using batch_covariance
    mean_est, cov_est = batch_covariance(batches)

    # Compute covariance directly using PyTorch
    mean_direct = data.mean(0)
    cov_direct = torch.cov(data.T)

    # Compare results
    assert torch.allclose(
        mean_est, mean_direct, atol=1e-6
    ), "Mean estimates do not match"
    assert torch.allclose(
        cov_est, cov_direct, atol=1e-6
    ), "Covariance estimates do not match"
