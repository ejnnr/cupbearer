import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cupbearer.detectors.anomaly_detector import ActivationBasedDetector
from cupbearer.detectors.mahalanobis.helpers import mahalanobis, update_covariance


class MahalanobisDetector(ActivationBasedDetector):
    def train(
        self,
        dataset,
        max_batches: int = 0,
        relative: bool = False,
        rcond: float = 1e-5,
        batch_size: int = 4096,
        pbar: bool = True,
        debug: bool = False,
    ):
        # This entire training method doesn't require gradients, we're just computing
        # averages etc.
        with torch.no_grad():
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            )
            example_batch = next(iter(data_loader))
            _, example_activations = self.get_activations(example_batch)

            # v is an entire batch, v[0] are activations for a single input
            activation_sizes = {k: v[0].numel() for k, v in example_activations.items()}
            means = {k: torch.zeros(size) for k, size in activation_sizes.items()}
            Cs = {k: torch.zeros((size, size)) for k, size in activation_sizes.items()}
            ns = {k: 0 for k in activation_sizes.keys()}

            # TODO: could consider computing a separate mean for each class,
            # I think that's more standard for OOD detection in classification,
            # but less general and maybe less analogous to the setting I care about.
            if pbar:
                data_loader = tqdm(data_loader)

            for i, batch in enumerate(data_loader):
                if max_batches and i >= max_batches:
                    break
                _, activations = self.get_activations(batch)

                for k, activation in activations.items():
                    # Flatten the activations to (batch, dim)
                    activation = activation.reshape(activation.shape[0], -1)
                    means[k], Cs[k], ns[k] = update_covariance(
                        means[k], Cs[k], ns[k], activation
                    )

            self.means = means
            self.covariances = {k: C / (ns[k] - 1) for k, C in Cs.items()}
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
