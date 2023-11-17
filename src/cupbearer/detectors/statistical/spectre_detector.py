import torch

from cupbearer.detectors.statistical.helpers import quantum_entropy
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


class SpectreDetector(ActivationCovarianceBasedDetector):
    def post_covariance_training(self, rcond: float):
        whitening_matrices = {}
        for k, cov in self.covariances.items():
            # Compute decomposition
            eigs = torch.linalg.eigh(cov)

            # Zero entries corresponding to eigenvalues smaller than rcond
            vals_rsqrt = eigs.eigenvalues.rsqrt()
            vals_rsqrt[eigs.eigenvalues < rcond * eigs.eigenvalues.max()] = 0

            # PCA whitening
            # following https://doi.org/10.1080/00031305.2016.1277159
            # and https://stats.stackexchange.com/a/594218/319192
            # but transposed (sphering with x@W instead of W@x)
            whitening_matrices[k] = eigs.eigenvectors * vals_rsqrt.unsqueeze(0)
            assert torch.allclose(
                whitening_matrices[k], eigs.eigenvectors @ vals_rsqrt.diag()
            )
        self.whitening_matrices = whitening_matrices

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
        return quantum_entropy(  # TODO should possibly pass rank
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
