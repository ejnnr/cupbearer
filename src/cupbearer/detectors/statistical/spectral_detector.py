import torch

from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


class SpectralSignatureDetector(ActivationCovarianceBasedDetector):
    """Detector based on "spectral signatures" as described in:

    Tran, Brandon et al. “Spectral Signatures in Backdoor Attacks.”
    Neural Information Processing Systems (2018).
    """

    def post_covariance_training(self, _):
        # Calculate top right singular vectors from covariance matrices
        self.top_singular_vectors = {
            k: torch.linalg.eigh(cov).eigenvectors[:, -1]
            for k, cov in self.covariances.items()
        }

    def layerwise_scores(self, batch):
        # ((R(x_i) - \hat{R}) * v) ** 2
        _, activations = self.get_activations(batch)
        outlier_scores = {
            k: torch.einsum(
                "bi,i->b",
                (activations[k] - self.means[k]),
                v,
            ).square()
            for k, v in self.top_singular_vectors.items()
        }
        return outlier_scores

    def _get_trained_variables(self, saving: bool = False):
        return {"means": self.means, "top_singular_vectors": self.top_singular_vectors}

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.top_singular_vectors = variables["top_singular_vectors"]
