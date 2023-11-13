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


@pytest.mark.parametrize("N", [15, 100])
def test_spectral_computation(N: int):
    # Create synthetic data with non-trivial covariance and 0 mean
    # torch.manual_seed(16380)
    mean = torch.tensor([0.0, 0.0, 0.0])
    cov = torch.tensor([[2.0, 1.0, 0.5], [1.0, 4.0, 0.0], [0.5, 0.0, 3.0]])
    dist = torch.distributions.MultivariateNormal(mean, cov)
    data = dist.sample((N,))
    data = data - data.mean(dim=0)

    # Compute svd
    usvh = torch.linalg.svd(data, full_matrices=False)
    assert usvh.S[0] == usvh.S.max()  # descending order
    v_direct = usvh.Vh[0, :]
    assert v_direct.numel() == data.size(1)
    assert v_direct.ndim == 1
    assert torch.allclose(usvh.Vh.square().sum(dim=0), torch.Tensor([1.0]))

    # Compute singular vectors via covariance matrix
    eigs = torch.linalg.eigh(torch.cov(data.T))
    assert eigs.eigenvalues[-1] == eigs.eigenvalues.max()  # ascending order
    v_indirect = eigs.eigenvectors[:, -1]

    # Check that variables are what we think they are
    assert torch.allclose(eigs.eigenvectors.square().sum(dim=1), torch.Tensor([1.0]))
    assert torch.allclose(eigs.eigenvectors.flip(-1).abs(), usvh.Vh.mH.abs())
    assert torch.allclose(
        eigs.eigenvalues.flip(0), usvh.S.square() / (data.size(0) - 1)
    )

    # Check what we actually care about
    try:
        assert torch.allclose(v_direct, v_indirect)
    except AssertionError:
        # Sign ambiguity
        assert torch.allclose(-v_direct, v_indirect)
