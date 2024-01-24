import torch

from cupbearer.detectors.statistical.helpers import mahalanobis
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
    MahalanobisTrainConfig,
)


class MahalanobisDetector(ActivationCovarianceBasedDetector):
    def post_covariance_training(self, train_config: MahalanobisTrainConfig):
        self.inv_covariances = {
            k: torch.linalg.pinv(C, rcond=train_config.rcond, hermitian=True)
            for k, C in self.covariances.items()
        }
        self.inv_diag_covariances = None
        if train_config.relative:
            self.inv_diag_covariances = {
                k: torch.where(torch.diag(C) > train_config.rcond, 1 / torch.diag(C), 0)
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
