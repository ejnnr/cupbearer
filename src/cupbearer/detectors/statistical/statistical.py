from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cupbearer.detectors.anomaly_detector import ActivationBasedDetector


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
