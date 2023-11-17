import warnings
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cupbearer.detectors.anomaly_detector import ActivationBasedDetector
from cupbearer.detectors.statistical.helpers import update_covariance


class StatisticalDetector(ActivationBasedDetector, ABC):
    @abstractmethod
    def init_variables(self, activation_sizes: dict[str, torch.Size]):
        pass

    @abstractmethod
    def batch_update(self, activations: dict[str, torch.Tensor]):
        pass

    def train(
        self,
        dataset,
        *,
        max_batches: int = 0,
        batch_size: int = 4096,
        pbar: bool = True,
        **kwargs,
    ):
        # Common for statistical methods is that the training does not require
        # gradients, but instead computes makes use of summary statistics or
        # similar
        with torch.inference_mode():
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            )
            example_batch = next(iter(data_loader))
            _, example_activations = self.get_activations(example_batch)

            # v is an entire batch, v[0] are activations for a single input
            activation_sizes = {k: v[0].size() for k, v in example_activations.items()}
            self.init_variables(activation_sizes)

            if pbar:
                data_loader = tqdm(data_loader)

            for i, batch in enumerate(data_loader):
                if max_batches and i >= max_batches:
                    break
                _, activations = self.get_activations(batch)
                self.batch_update(activations)


class ActivationCovarianceBasedDetector(StatisticalDetector):
    """Generic abstract detector that learns means and covariance matrices
    during training."""

    def init_variables(self, activation_sizes: dict[str, torch.Size]):
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
            activation = activation.flatten(start_dim=1)
            self._means[k], self._Cs[k], self._ns[k] = update_covariance(
                self._means[k], self._Cs[k], self._ns[k], activation
            )

    @abstractmethod
    def post_covariance_training(self, rcond: float):
        pass

    def train(
        self,
        dataset,
        *,
        max_batches: int = 0,
        rcond: float = 1e-5,
        batch_size: int = 4096,
        pbar: bool = True,
        **kwargs,
    ):
        super().train(
            dataset, max_batches=max_batches, batch_size=batch_size, pbar=pbar
        )

        # Post process
        with torch.inference_mode():
            self.means = self._means
            self.covariances = {k: C / (self._ns[k] - 1) for k, C in self._Cs.items()}
            has_full_rank = {
                k: torch.linalg.matrix_rank(cov) == cov.size(0)
                for k, cov in self.covariances.items()
            }
            if not all(has_full_rank.items()):
                warnings.warn(
                    "Only {sum(has_full_rank.values()) / len(has_full_rank)} layers "
                    "have full rank covariance matrices."
                )
            self.post_covariance_training(rcond=rcond)
