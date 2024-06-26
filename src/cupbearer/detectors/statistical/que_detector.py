import torch

from cupbearer.detectors.statistical.helpers import quantum_entropy
from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


class QuantumEntropyDetector(ActivationCovarianceBasedDetector):
    """Detector based on the "quantum entropy" score.

    Based on https://arxiv.org/abs/1906.11366 and inspired by SPECTRE
    (https://arxiv.org/abs/2104.11315) but much simpler. We don't do dimensionality
    reduction, and instead of using robust estimation for the clean mean and covariance,
    we just assume access to clean data like for our other anomaly detection methods.
    """

    use_untrusted: bool = True

    def post_covariance_training(self, rcond: float = 1e-5, **kwargs):
        whitening_matrices = {}
        for k, cov in self.covariances["trusted"].items():
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
        self.trusted_whitening_matrices = whitening_matrices

        self.untrusted_covariance_norms = {}
        for k, cov in self.covariances["untrusted"].items():
            self.untrusted_covariance_norms[k] = torch.linalg.eigvalsh(cov).max()

    def _individual_layerwise_score(self, name, activation):
        whitened_activations = torch.einsum(
            "bi,ij->bj",
            activation.flatten(start_dim=1) - self.means["trusted"][name],
            self.trusted_whitening_matrices[name],
        )
        # TODO should possibly pass rank
        return quantum_entropy(
            whitened_activations,
            self.covariances["untrusted"][name],
            self.untrusted_covariance_norms[name],
        )

    def _get_trained_variables(self, saving: bool = False):
        return {
            "means": self.means,
            "whitening_matrices": self.trusted_whitening_matrices,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.trusted_whitening_matrices = variables["whitening_matrices"]
