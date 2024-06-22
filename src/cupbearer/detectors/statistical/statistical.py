from abc import ABC, abstractmethod

import torch
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from cupbearer.detectors.activation_based import ActivationBasedDetector
from cupbearer.detectors.statistical.helpers import update_covariance


class StatisticalDetector(ActivationBasedDetector, ABC):
    use_trusted: bool = True

    @abstractmethod
    def init_variables(self, activation_sizes: dict[str, torch.Size], device):
        pass

    @abstractmethod
    def batch_update(self, activations: dict[str, torch.Tensor]):
        pass

    def train(
        self,
        trusted_data,
        untrusted_data,
        *,
        batch_size: int = 1024,
        pbar: bool = True,
        max_steps: int | None = None,
        **kwargs,
    ):
        # Common for statistical methods is that the training does not require
        # gradients, but instead computes summary statistics or similar
        with torch.inference_mode():
            if self.use_trusted:
                if trusted_data is None:
                    raise ValueError(
                        f"{self.__class__.__name__} requires trusted training data."
                    )
                data = trusted_data
            else:
                if untrusted_data is None:
                    raise ValueError(
                        f"{self.__class__.__name__} requires untrusted training data."
                    )
                data = untrusted_data

            # No reason to shuffle, we're just computing statistics
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
            example_batch = next(iter(data_loader))
            example_activations = self.get_activations(example_batch)

            # v is an entire batch, v[0] are activations for a single input
            activation_sizes = {k: v[0].size() for k, v in example_activations.items()}
            self.init_variables(
                activation_sizes, device=next(iter(example_activations.values())).device
            )

            if pbar:
                data_loader = tqdm(data_loader, total=max_steps or len(data_loader))

            for i, batch in enumerate(data_loader):
                if max_steps and i >= max_steps:
                    break
                activations = self.get_activations(batch)
                self.batch_update(activations)


class ActivationCovarianceBasedDetector(StatisticalDetector):
    """Generic abstract detector that learns means and covariance matrices
    during training."""

    def init_variables(self, activation_sizes: dict[str, torch.Size], device):
        if any(len(size) != 1 for size in activation_sizes.values()):
            logger.debug(
                "Received multi-dimensional activations, will only learn "
                "covariances along last dimension and treat others independently. "
                "If this is unintentional, pass "
                "`activation_preprocessing_func=utils.flatten_last`."
            )
        logger.debug(
            "Activation sizes: \n"
            + "\n".join(f"{k}: {size}" for k, size in activation_sizes.items())
        )
        self._means = {
            k: torch.zeros(size[-1], device=device)
            for k, size in activation_sizes.items()
        }
        self._Cs = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
        self._ns = {k: 0 for k in activation_sizes.keys()}

    def batch_update(self, activations: dict[str, torch.Tensor]):
        for k, activation in activations.items():
            # Flatten the activations to (batch, dim)
            activation = rearrange(activation, "batch ... dim -> (batch ...) dim")
            assert activation.ndim == 2, activation.shape
            self._means[k], self._Cs[k], self._ns[k] = update_covariance(
                self._means[k], self._Cs[k], self._ns[k], activation
            )

    @abstractmethod
    def post_covariance_training(self, **kwargs):
        pass

    @abstractmethod
    def _individual_layerwise_score(self, name: str, activation: torch.Tensor):
        """Compute the anomaly score for a single layer.

        `name` is passed in to access the mean/covariance and any custom derived
        quantities computed in post_covariance_training.

        `activation` will always have shape (batch, dim). The `batch` dimension might
        not just be the actual batch dimension, but could also contain multiple entries
        from a single sample, in the case of multi-dimensional activations that we
        treat as independent along all but the last dimension.

        Should return a tensor of shape (batch,) with the anomaly scores.
        """
        pass

    def layerwise_scores(self, batch):
        activations = self.get_activations(batch)
        batch_size = next(iter(activations.values())).shape[0]
        activations = {
            k: rearrange(v, "batch ... dim -> (batch ...) dim")
            for k, v in activations.items()
        }
        scores = {
            k: self._individual_layerwise_score(k, v) for k, v in activations.items()
        }
        scores = {
            k: rearrange(
                v,
                "(batch independent_dims) -> batch independent_dims",
                batch=batch_size,
            ).sum(-1)
            for k, v in scores.items()
        }
        return scores

    def train(self, trusted_data, untrusted_data, **kwargs):
        super().train(
            trusted_data=trusted_data, untrusted_data=untrusted_data, **kwargs
        )

        # Post process
        with torch.inference_mode():
            self.means = self._means
            self.covariances = {k: C / (self._ns[k] - 1) for k, C in self._Cs.items()}
            if any(torch.count_nonzero(C) == 0 for C in self.covariances.values()):
                raise RuntimeError("All zero covariance matrix detected.")

            self.post_covariance_training(**kwargs)
