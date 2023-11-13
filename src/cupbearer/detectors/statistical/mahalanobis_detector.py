import torch

from cupbearer.detectors.statistical.helpers import mahalanobis, update_covariance
from cupbearer.detectors.statistical.statistical import StatisticalDetector


class MahalanobisDetector(StatisticalDetector):
    def init_variables(self, activation_sizes: dict[str, torch.Size]):
        # TODO: could consider computing a separate mean for each class,
        # I think that's more standard for OOD detection in classification,
        # but less general and maybe less analogous to the setting I care about.
        self._means = {
            k: torch.zeros(size.numel()) for k, size in activation_sizes.items()
        }
        self._Cs = {
            k: torch.zeros((size.numel(), size.numel()))
            for k, size in activation_sizes.items()
        }
        self._ns = {k: 0 for k in activation_sizes.keys()}

    def batch_update(self, activations: dict[str, torch.Tensor]):
        for k, activation in activations.items():
            # Flatten the activations to (batch, dim)
            activation = activation.reshape(activation.shape[0], -1)
            self._means[k], self._Cs[k], self._ns[k] = update_covariance(
                self._means[k], self._Cs[k], self._ns[k], activation
            )

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
            dataset, max_batches=max_batches, batch_size=batch_size, pbar=pbar
        )

        # Post process
        self.means = self._means
        self.covariances = {k: C / (self._ns[k] - 1) for k, C in self._Cs.items()}
        self.inv_covariances = {
            k: torch.linalg.pinv(C, rcond=rcond, hermitian=True)
            for k, C in self.covariances.items()
        }
        self.inv_diag_covariances = None
        if relative:
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
