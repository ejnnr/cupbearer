import torch

from cupbearer.detectors.statistical.helpers import mahalanobis
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


def _pinv(C, rcond, dtype=torch.float64):
    # Workaround for pinv not being supported on MPS
    if C.is_mps:
        return (
            torch.linalg.pinv(C.cpu().to(dtype), rcond=rcond, hermitian=True)
            .to(C.dtype)
            .to(C.device)
        )
    return torch.linalg.pinv(C.to(dtype), rcond=rcond, hermitian=True).to(C.dtype)


class MahalanobisDetector(ActivationCovarianceBasedDetector):
    def post_covariance_training(
        self,
        rcond: float = 1e-5,
        relative: bool = False,
        shrinkage: float = 0.0,
        **kwargs,
    ):
        covariances = self.covariances["trusted"]
        self.inv_covariances = {}
        for k, C in covariances.items():
            if shrinkage > 0:
                C = (1 - shrinkage) * C + shrinkage * torch.trace(C).mean() * torch.eye(
                    C.shape[0], dtype=C.dtype, device=C.device
                )
            self.inv_covariances[k] = _pinv(C, rcond)

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

    def _get_trained_variables(self):
        return {
            "means": self.means,
            "inv_covariances": self.inv_covariances,
            "inv_diag_covariances": self.inv_diag_covariances,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.inv_covariances = variables["inv_covariances"]
        self.inv_diag_covariances = variables["inv_diag_covariances"]

    def __repr__(self):
        return "MahalanobisDetector()"
