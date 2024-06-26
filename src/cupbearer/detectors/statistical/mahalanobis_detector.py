import torch

from cupbearer.detectors.statistical.helpers import mahalanobis
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


def _pinv(C, rcond):
    # Workaround for pinv not being supported on MPS
    if C.is_mps:
        return torch.linalg.pinv(C.cpu(), rcond=rcond, hermitian=True).to(C.device)
    return torch.linalg.pinv(C, rcond=rcond, hermitian=True)


class MahalanobisDetector(ActivationCovarianceBasedDetector):
    def post_covariance_training(
        self, rcond: float = 1e-5, relative: bool = False, **kwargs
    ):
        self.inv_covariances = {
            k: _pinv(C, rcond) for k, C in self.covariances["trusted"].items()
        }
        self.inv_diag_covariances = None
        if relative:
            self.inv_diag_covariances = {
                k: torch.where(torch.diag(C) > rcond, 1 / torch.diag(C), 0)
                for k, C in self.covariances["trusted"].items()
            }

    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        inv_diag_covariance = None
        if self.inv_diag_covariances is not None:
            inv_diag_covariance = self.inv_diag_covariances[name]

        distance = mahalanobis(
            activation,
            self.means["trusted"][name],
            self.inv_covariances[name],
            inv_diag_covariance=inv_diag_covariance,
        )

        # Normalize by the number of dimensions (no sqrt since we're using *squared*
        # Mahalanobis distance)
        return distance / self.means["trusted"][name].shape[0]

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
