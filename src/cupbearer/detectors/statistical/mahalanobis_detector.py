import numpy as np
import torch

from cupbearer.detectors.statistical.helpers import mahalanobis
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


def _pinv(C, rcond):
    # Workaround for pinv not being supported on MPS
    if C.device.type == "mps":
        return torch.linalg.pinv(C.cpu(), rcond=rcond, hermitian=True).to(C.device)
    return torch.linalg.pinv(C, rcond=rcond, hermitian=True)


# From https://gist.github.com/chausies/011df759f167b17b5278264454fff379
def norm_cdf(x):
    return (1 + torch.erf(x / np.sqrt(2))) / 2


def log_norm_cdf_helper(x):
    a = 0.344
    b = 5.334
    return ((1 - a) * x + a * x**2 + b).sqrt()


def log_norm_cdf(x):
    thresh = 3
    out = x * 0
    low = x < -thresh
    high = x > thresh
    middle = torch.logical_and(x >= -thresh, x <= thresh)
    out[middle] = norm_cdf(x[middle]).log()
    out[low] = -(
        (x[low] ** 2 + np.log(2 * np.pi)) / 2 + log_norm_cdf_helper(-x[low]).log()
    )
    out[high] = torch.log1p(
        -(-(x[high] ** 2) / 2).exp() / np.sqrt(2 * np.pi) / log_norm_cdf_helper(x[high])
    )
    return out


def log_chi_squared_percentiles(distances, dim):
    # Approximate chi^2 with dim degrees of freedom with a standard normal with mean
    # dim and variance 2 * dim. This is a good approximation for large dim.
    x = (distances - dim) / np.sqrt(2 * dim)
    # log_standard_normal_cdf returns the log of P(X < x),
    # but we want P(X > x) = P(X < -x)
    return -log_norm_cdf(-x)


class MahalanobisDetector(ActivationCovarianceBasedDetector):
    def post_covariance_training(
        self, rcond: float = 1e-5, relative: bool = False, **kwargs
    ):
        self.inv_covariances = {k: _pinv(C, rcond) for k, C in self.covariances.items()}
        self.inv_diag_covariances = None
        if relative:
            self.inv_diag_covariances = {
                k: torch.where(torch.diag(C) > rcond, 1 / torch.diag(C), 0)
                for k, C in self.covariances.items()
            }

    def layerwise_scores(self, batch):
        activations = self.get_activations(batch)
        distances = mahalanobis(
            activations,
            self.means,
            self.inv_covariances,
            inv_diag_covariances=self.inv_diag_covariances,
        )
        dims = {k: v.shape[0] for k, v in self.means.items()}
        return {
            k: log_chi_squared_percentiles(v, dims[k]) for k, v in distances.items()
        }

    def _get_trained_variables(self, saving: bool = False):
        return {
            "means": self.means,
            "inv_covariances": self.inv_covariances,
            "inv_diag_covariances": self.inv_diag_covariances,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.inv_covariances = variables["inv_covariances"]
        self.inv_diag_covariances = variables["inv_diag_covariances"]
