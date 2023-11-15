import torch

from cupbearer.detectors.statistical.helpers import mahalanobis
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


class MahalanobisDetector(ActivationCovarianceBasedDetector):
    def post_covariance_training(self, rcond, relative: bool = False, **kwargs):
        self.inv_covariances = {
            k: torch.linalg.pinv(C, rcond=rcond, hermitian=True)
            for k, C in self.covariances.items()
        }

    def train(
        self,
        dataset,
        *,
        max_batches: int = 0,
        relative: bool = False,
        rcond: float = 1e-5,
        batch_size: int = 4096,
        pbar: bool = True,
        **kwargs,
    ):
        super().train(
            dataset,
            max_batches=max_batches,
            rcond=rcond,
            batch_size=batch_size,
            pbar=pbar,
            **kwargs,
        )
        self.inv_diag_covariances = None
        if relative:
            with torch.inference_mode():
                self.inv_diag_covariances = {
                    k: torch.where(torch.diag(C) > rcond, 1 / torch.diag(C), 0)
                    for k, C in self.covariances.items()
                }

    def layerwise_scores(self, batch):
        _, activations = self.get_activations(batch)
        return mahalanobis(
            activations,
            self.means,
            self.inv_covariances,
            inv_diag_covariances=self.inv_diag_covariances,
        )

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
