import torch

from cupbearer.detectors.statistical.statistical import (
    ActivationCovarianceBasedDetector,
)


class SpectralSignatureDetector(ActivationCovarianceBasedDetector):
    """Detector based on "spectral signatures" as described in:

    Tran, Brandon et al. “Spectral Signatures in Backdoor Attacks.”
    Neural Information Processing Systems (2018).
    """

    use_trusted: bool = False
    use_untrusted: bool = True

    def post_covariance_training(self, **kwargs):
        # Calculate top right singular vectors from covariance matrices
        self.top_singular_vectors = {
            k: torch.linalg.eigh(cov).eigenvectors[:, -1]
            for k, cov in self.covariances["untrusted"].items()
        }

    def _individual_layerwise_score(self, name, activation):
        # ((R(x_i) - \hat{R}) * v) ** 2
        return torch.einsum(
            "bi,i->b",
            (activation - self.means["untrusted"][name]),
            self.top_singular_vectors[name],
        ).square()

    def _get_trained_variables(self, saving: bool = False):
        return {"means": self.means, "top_singular_vectors": self.top_singular_vectors}

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.top_singular_vectors = variables["top_singular_vectors"]
