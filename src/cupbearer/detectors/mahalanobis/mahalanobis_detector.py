import jax.numpy as jnp
from torch.utils.data import DataLoader
from tqdm import tqdm

from cupbearer.data import numpy_collate
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.detectors.mahalanobis.helpers import mahalanobis, update_covariance


class MahalanobisDetector(AnomalyDetector):
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
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=numpy_collate,
        )
        example_inputs = next(iter(data_loader))
        _, example_activations = self._model(example_inputs)
        # For each layer, get the number of entries of activations (without batch dim)
        activation_sizes = [x[0].size for x in example_activations]
        means = [jnp.zeros(size) for size in activation_sizes]
        Cs = [jnp.zeros((size, size)) for size in activation_sizes]
        ns = [0 for _ in activation_sizes]

        # TODO: could consider computing a separate mean for each class,
        # I think that's more standard for OOD detection in classification,
        # but less general and maybe less analogous to the setting I care about.
        if pbar:
            data_loader = tqdm(data_loader)

        for i, batch in enumerate(data_loader):
            if max_batches and i >= max_batches:
                break
            _, activations = self._model(batch)
            # TODO: use jit compilation
            for i, activation in enumerate(activations):
                # Flatten the activations to (batch, dim)
                activation = activation.reshape(activation.shape[0], -1)
                means[i], Cs[i], ns[i] = update_covariance(
                    means[i], Cs[i], ns[i], activation
                )

        self.means = means
        self.covariances = [C / (n - 1) for C, n in zip(Cs, ns)]
        self.inv_covariances = [
            jnp.linalg.pinv(C, rcond=rcond, hermitian=True) for C in self.covariances
        ]
        self.inv_diag_covariances = None
        if relative:
            self.inv_diag_covariances = [
                jnp.where(jnp.diag(C) > rcond, 1 / jnp.diag(C), 0)
                for C in self.covariances
            ]

    def layerwise_scores(self, batch):
        _, activations = self._model(batch)
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
