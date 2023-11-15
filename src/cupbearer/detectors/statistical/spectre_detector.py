import torch

from cupbearer.detectors.statistical.helpers import quantum_entropy
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


class SpectreDetector(ActivationCovarianceBasedDetector):
    def post_covariance_training(self, rcond: float):
        whitening_matrices = {}
        for k, cov in self.covariances.items():
            inv_cov = torch.linalg.pinv(cov, rcond=rcond, hermitian=True)
            assert inv_cov.ndim == 2
            # Cholesky whitening, could use PCA or ZCA instead
            whitening_matrices[k] = torch.linalg.cholesky(inv_cov)
            # following https://doi.org/10.1080/00031305.2016.1277159
            # but moving transpose to einsum
            # TODO move this assertion to tests/
            assert torch.allclose(
                torch.einsum(
                    "ik,km->im",
                    torch.einsum(
                        "ij,kj->ik",
                        whitening_matrices[k],
                        whitening_matrices[k],
                    ),
                    cov,
                ),
                torch.eye(n=cov.size(0)),
            )

    def layerwise_scores(self, batch):
        _, activations = self.get_activations(batch)
        whitened_activations = {
            k: torch.einsum(
                "bi,ij->bj",
                activations[k].flatten(start_dim=1) - self.means[k],
                self.whitening_matrices[k],
            )
            for k in self.activations
        }
        return quantum_entropy(
            whitened_activations,
        )

    def _get_trained_variables(self, saving: bool = False):
        return {
            "means": self.means,
            "whitening_matrices": self.whitening_matrices,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.whitening_matrices = variables["whitening_matrices"]
