from pathlib import Path
import sys
from typing import Optional
import hydra
from hydra.utils import to_absolute_path
import jax
import jax.numpy as jnp
import flax.linen as nn
from torch.utils.data import DataLoader
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
import sklearn.metrics
from tqdm import tqdm

from abstractions import abstraction, data, utils
from abstractions.adversarial_examples import AdversarialExampleDataset
from abstractions.anomaly_detector import AnomalyDetector
from abstractions.computations import get_abstraction_maps


def update_covariance(curr_mean, curr_C, curr_n, new_data):
    # Should be (batch, dim)
    assert new_data.ndim == 2

    new_n = len(new_data)
    total_n = curr_n + new_n

    new_mean = jnp.mean(new_data, axis=0)
    delta_mean = new_mean - curr_mean
    updated_mean = (curr_n * curr_mean + new_n * new_mean) / total_n

    delta_data = new_data - new_mean
    new_C = jnp.dot(delta_data.T, delta_data)
    updated_C = (
        curr_C + new_C + curr_n * new_n * jnp.outer(delta_mean, delta_mean) / total_n
    )

    return updated_mean, updated_C, total_n


def batch_covariance(batches):
    mean = jnp.zeros(batches[0].shape[1])
    C = jnp.zeros((batches[0].shape[1], batches[0].shape[1]))
    n = 0

    for batch in batches:
        mean, C, n = update_covariance(mean, C, n, batch)

    return mean, C / (n - 1)  # Apply Bessel's correction for sample covariance


def mahalanobis(
    activations: list[jax.Array],
    means: list[jax.Array],
    inv_covariances: list[jax.Array],
    inv_diag_covariances: Optional[list[jax.Array]] = None,
):
    """Compute Simplified Relative Mahalanobis distances for a batch of activations.

    The Mahalanobis distance for each layer is computed, and the distances are then averaged
    over layers.

    Args:
        activations: List of activations for each layer, each element has shape (batch, dim)
        means: List of means for each layer, each element has shape (dim,)
        inv_covariances: List of inverse covariances for each layer, each element has shape (dim, dim)
        inv_diag_covariances: List of inverse diagonal covariances for each layer, each element has shape (dim,).
            If None, the usual Mahalanobis distance is computed instead of the
            (simplified) relative Mahalanobis distance.

    Returns:
        Mahalanobis distance for each element in the batch, shape (batch,)
    """
    batch_size = activations[0].shape[0]
    distances: list[jax.Array] = []
    for i, activation in enumerate(activations):
        activation = activation.reshape(batch_size, -1)
        delta = activation - means[i]
        assert delta.ndim == 2 and delta.shape[0] == batch_size
        distance = jnp.sum((delta @ inv_covariances[i]) * delta, axis=1)
        if inv_diag_covariances is not None:
            distance -= jnp.sum(delta**2 * inv_diag_covariances[i][None], axis=1)
        distances.append(distance)
    return jnp.array(distances)


class MahalanobisDetector(AnomalyDetector):
    def train(
        self,
        dataset,
        max_batches: int = 0,
        relative: bool = False,
        rcond: float = 1e-5,
        batch_size: int = 4096,
        pbar: bool = True,
    ):
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data.numpy_collate,
        )
        example_inputs = next(iter(data_loader))
        _, example_activations = self._model(example_inputs)
        # For each layer, get the number of elements of the activations (without batch dim)
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

    def _get_trained_variables(self):
        return {
            "means": self.means,
            "inv_covariances": self.inv_covariances,
            "inv_diag_covariances": self.inv_diag_covariances,
        }

    def _set_trained_variables(self, variables):
        self.means = variables["means"]
        self.inv_covariances = variables["inv_covariances"]
        self.inv_diag_covariances = variables["inv_diag_covariances"]


@hydra.main(version_base=None, config_path="conf", config_name="mahalanobis")
def main(cfg: DictConfig):
    """Execute model training and evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    # Load the full model we want to abstract
    base_run = Path(cfg.base_run)
    base_cfg = OmegaConf.load(
        to_absolute_path(str(base_run / ".hydra" / "config.yaml"))
    )

    full_computation = hydra.utils.call(base_cfg.model)
    full_model = abstraction.Model(computation=full_computation)
    full_params = utils.load(to_absolute_path(str(base_run / "model.pytree")))["params"]

    detector = MahalanobisDetector(
        model=full_model,
        params=full_params,
        max_batch_size=cfg.max_batch_size,
    )

    train_dataset = data.get_dataset(base_cfg.train_data)

    detector.train(
        train_dataset,
        max_batches=cfg.max_batches,
        relative=cfg.relative,
        rcond=cfg.rcond,
        batch_size=cfg.batch_size,
        pbar=cfg.pbar,
    )
    detector.save("detector")


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, color="<green>", icon="📈")
    logger.add(
        sys.stderr, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stderr, filter=lambda record: record["level"].name != "METRICS")
    # We want to escape slashes in arguments that get reused as filenames.
    OmegaConf.register_new_resolver("escape", lambda x: x.replace("/", "_"))
    main()
