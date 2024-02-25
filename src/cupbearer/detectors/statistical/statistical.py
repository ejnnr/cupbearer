from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cupbearer.detectors.anomaly_detector import ActivationBasedDetector
from cupbearer.detectors.statistical.helpers import update_covariance
from cupbearer.utils.utils import BaseConfig


@dataclass
class StatisticalTrainConfig(BaseConfig, ABC):
    max_batches: int = 0
    batch_size: int = 4096
    max_batch_size: int = 4096
    pbar: bool = True
    num_workers: int = 0
    # robust: bool = False  # TODO spectre uses
    # https://www.semanticscholar.org/paper/Being-Robust-(in-High-Dimensions)-Can-Be-Practical-Diakonikolas-Kamath/2a6de51d86f13e9eb7efa85491682dad0ccd65e8?utm_source=direct_link

    def get_dataloader(self, dataset, train=True):
        if train:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )


@dataclass
class DebugStatisticalTrainConfig(StatisticalTrainConfig):
    max_batchs: int = 3
    batch_size: int = 5
    max_batch_size: int = 5


@dataclass
class ActivationCovarianceTrainConfig(StatisticalTrainConfig):
    rcond: float = 1e-5


@dataclass
class DebugActivationCovarianceTrainConfig(
    DebugStatisticalTrainConfig, ActivationCovarianceTrainConfig
):
    pass


@dataclass
class MahalanobisTrainConfig(ActivationCovarianceTrainConfig):
    relative: bool = False


@dataclass
class DebugMahalanobisTrainConfig(DebugStatisticalTrainConfig, MahalanobisTrainConfig):
    pass


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
        num_classes: int,
        train_config: StatisticalTrainConfig,
    ):
        # Common for statistical methods is that the training does not require
        # gradients, but instead computes summary statistics or similar
        with torch.inference_mode():
            data_loader = train_config.get_dataloader(dataset)
            example_batch = next(iter(data_loader))
            _, example_activations = self.get_activations(example_batch)

            # v is an entire batch, v[0] are activations for a single input
            activation_sizes = {k: v[0].size() for k, v in example_activations.items()}
            self.init_variables(activation_sizes)

            if train_config.pbar:
                data_loader = tqdm(data_loader)

            for i, batch in enumerate(data_loader):
                if train_config.max_batches and i >= train_config.max_batches:
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
    def post_covariance_training(self, train_config: ActivationCovarianceTrainConfig):
        pass

    def train(
        self,
        dataset,
        *,
        num_classes: int,
        train_config: ActivationCovarianceTrainConfig,
    ):
        super().train(
            dataset,
            num_classes=num_classes,
            train_config=train_config,
        )

        # Post process
        with torch.inference_mode():
            self.means = self._means
            self.covariances = {k: C / (self._ns[k] - 1) for k, C in self._Cs.items()}
            if any(torch.count_nonzero(C) == 0 for C in self.covariances.values()):
                raise RuntimeError("All zero covariance matrix detected.")

            self.post_covariance_training(train_config=train_config)
