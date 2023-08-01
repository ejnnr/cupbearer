import math

import jax.numpy as jnp
import numpy as np
import pytest
from cupbearer.detectors.mahalanobis.helpers import batch_covariance


@pytest.mark.parametrize("N", [10, 15, 100])
def test_batch_covariance(N):
    # Create synthetic data with non-trivial covariance
    np.random.seed(42)
    mean = np.array([1.0, -2.0, 3.0])
    cov = np.array([[2.0, 1.0, 0.5], [1.0, 4.0, 0.0], [0.5, 0.0, 3.0]])
    data = np.random.multivariate_normal(mean, cov, size=N)

    # Split data into batches
    batch_size = 9
    # Round up, so we'll have a final batch with smaller size to also test that case
    num_batches = math.ceil(len(data) / batch_size)
    batches = [
        jnp.array(data[i * batch_size : (i + 1) * batch_size])
        for i in range(num_batches)
    ]

    # Compute covariance using batch_covariance
    mean_est, cov_est = batch_covariance(batches)

    # Compute covariance directly using JAX
    mean_direct = jnp.mean(data, axis=0)
    cov_direct = jnp.cov(data, rowvar=False)

    # Compare results
    assert jnp.allclose(mean_est, mean_direct, atol=1e-6), "Mean estimates do not match"
    assert jnp.allclose(
        cov_est, cov_direct, atol=1e-6
    ), "Covariance estimates do not match"
