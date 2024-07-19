from dataclasses import dataclass
from typing import Any, Callable

import torch
import tqdm
from einops import rearrange
from loguru import logger

from cupbearer import utils
from cupbearer.data import MixedData
from cupbearer.detectors.activation_based import ActivationCache
from cupbearer.detectors.statistical.helpers import update_covariance
from cupbearer.tasks import Task


class StatisticsCollector:
    # TODO: this is just copied from ActivationBasedDetector, should be refactored
    def __init__(
        self,
        activation_names: list[str],
        activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        cache: ActivationCache | None = None,
    ):
        self.activation_names = activation_names
        self.activation_processing_func = activation_processing_func
        self.cache = cache

    def set_model(self, model: torch.nn.Module):
        # This is separate from __init__ because we want to be able to set the model
        # automatically based on the task, instead of letting the user pass it in.
        # On the other hand, it's separate from train() because we might need to set
        # the model even when just using the detector for inference.
        #
        # Subclasses can implement more complex logic here.
        self.model = model

    def _get_activations_no_cache(self, inputs) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        inputs = utils.inputs_to_device(inputs, device)
        acts = utils.get_activations(self.model, self.activation_names, inputs)

        # Can be used to for example select activations at specific token positions
        if self.activation_processing_func is not None:
            acts = {
                k: self.activation_processing_func(v, inputs, k)
                for k, v in acts.items()
            }

        return acts

    def get_activations(self, batch) -> dict[str, torch.Tensor]:
        inputs = utils.inputs_from_batch(batch)

        if self.cache is None:
            return self._get_activations_no_cache(inputs)

        return self.cache.get_activations(
            inputs, self.activation_names, self._get_activations_no_cache
        )

    def init_variables(self, activation_sizes: dict[str, torch.Size], device):
        self.between_class_variances = {
            k: torch.tensor(0.0, device=device) for k in self.activation_names
        }
        if any(len(size) != 1 for size in activation_sizes.values()):
            logger.debug(
                "Received multi-dimensional activations, will only learn "
                "covariances along last dimension and treat others independently. "
                "If this is unintentional, pass "
                "`activation_preprocessing_func=utils.flatten_last`."
            )
        self.means = {
            k: torch.zeros(size[-1], device=device)
            for k, size in activation_sizes.items()
        }
        self.normal_means = {
            k: torch.zeros(size[-1], device=device)
            for k, size in activation_sizes.items()
        }
        self.anomalous_means = {
            k: torch.zeros(size[-1], device=device)
            for k, size in activation_sizes.items()
        }
        # These are not the actual covariance matrices, they're missing a normalization
        # factor that we apply at the end.
        self._Cs = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
        self._normal_Cs = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
        self._anomalous_Cs = {
            k: torch.zeros((size[-1], size[-1]), device=device)
            for k, size in activation_sizes.items()
        }
        self._ns = {k: 0 for k in activation_sizes.keys()}
        self._normal_ns = {k: 0 for k in activation_sizes.keys()}
        self._anomalous_ns = {k: 0 for k in activation_sizes.keys()}

    def batch_update(self, activations: dict[str, torch.Tensor], labels: torch.Tensor):
        assert labels.ndim == 1
        labels = labels.bool()

        for k, activation in activations.items():
            # Flatten the activations to (batch, dim)
            normal_activation = rearrange(
                activation[~labels], "batch ... dim -> (batch ...) dim"
            )
            anomalous_activation = rearrange(
                activation[labels], "batch ... dim -> (batch ...) dim"
            )
            activation = rearrange(activation, "batch ... dim -> (batch ...) dim")

            # Update covariances and means
            self.means[k], self._Cs[k], self._ns[k] = update_covariance(
                self.means[k], self._Cs[k], self._ns[k], activation
            )

            if normal_activation.shape[0] > 0:
                (
                    self.normal_means[k],
                    self._normal_Cs[k],
                    self._normal_ns[k],
                ) = update_covariance(
                    self.normal_means[k],
                    self._normal_Cs[k],
                    self._normal_ns[k],
                    normal_activation,
                )

            if anomalous_activation.shape[0] > 0:
                (
                    self.anomalous_means[k],
                    self._anomalous_Cs[k],
                    self._anomalous_ns[k],
                ) = update_covariance(
                    self.anomalous_means[k],
                    self._anomalous_Cs[k],
                    self._anomalous_ns[k],
                    anomalous_activation,
                )

    def train(
        self,
        data: MixedData,
        *,
        batch_size: int = 1024,
        pbar: bool = True,
        max_steps: int | None = None,
    ):
        # Adapted from StatisticalDetector.train
        # TODO: figure out a way to refactor

        assert isinstance(data, MixedData)
        assert data.return_anomaly_labels

        with torch.inference_mode():
            data_loader = torch.utils.data.DataLoader(
                data, batch_size=batch_size, shuffle=True
            )
            example_batch, example_labels = next(iter(data_loader))
            example_activations = self.get_activations(example_batch)

            # v is an entire batch, v[0] are activations for a single input
            activation_sizes = {k: v[0].size() for k, v in example_activations.items()}
            self.init_variables(
                activation_sizes, device=next(iter(example_activations.values())).device
            )

            if pbar:
                data_loader = tqdm.tqdm(
                    data_loader, total=max_steps or len(data_loader)
                )

            for i, (batch, labels) in enumerate(data_loader):
                if max_steps and i >= max_steps:
                    break
                activations = self.get_activations(batch)
                self.batch_update(activations, labels)

        # Post processing for covariance
        with torch.inference_mode():
            self.covariances = {k: C / (self._ns[k] - 1) for k, C in self._Cs.items()}
            if any(torch.count_nonzero(C) == 0 for C in self.covariances.values()):
                raise RuntimeError("All zero covariance matrix detected.")

            self.normal_covariances = {
                k: C / (self._normal_ns[k] - 1) for k, C in self._normal_Cs.items()
            }
            self.anomalous_covariances = {
                k: C / (self._anomalous_ns[k] - 1)
                for k, C in self._anomalous_Cs.items()
            }

            self.total_variances = {
                k: self.covariances[k].trace() for k in self.activation_names
            }
            self.normal_variances = {
                k: self.normal_covariances[k].trace() for k in self.activation_names
            }
            self.anomalous_variances = {
                k: self.anomalous_covariances[k].trace() for k in self.activation_names
            }

            self.within_class_variances = {
                k: (
                    self._normal_ns[k] * self.normal_variances[k]
                    + self._anomalous_ns[k] * self.anomalous_variances[k]
                )
                / self._ns[k]
                for k in self.activation_names
            }

            self.between_class_variances = {
                k: self.total_variances[k] - self.within_class_variances[k]
                for k in self.activation_names
            }


@dataclass
class TaskData:
    activations: dict[str, torch.Tensor]
    labels: torch.Tensor
    collector: StatisticsCollector

    @staticmethod
    def from_task(
        task: Task,
        activation_names: list[str],
        n_samples: int = 64,
        activation_preprocessing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        cache: ActivationCache | None = None,
    ):
        collector = StatisticsCollector(
            activation_names=activation_names,
            activation_processing_func=activation_preprocessing_func,
            cache=cache,
        )
        collector.set_model(task.model)

        dataloader = torch.utils.data.DataLoader(
            task.test_data, batch_size=n_samples, shuffle=True
        )
        batch, labels = next(iter(dataloader))
        activations = collector.get_activations(batch)

        collector.train(data=task.test_data)

        return TaskData(
            activations=activations,
            labels=labels,
            collector=collector,
        )


def top_eigenvectors(matrix: torch.Tensor, n: int):
    mps = False
    if matrix.is_mps:
        mps = True
        matrix = matrix.cpu()
    eig = torch.linalg.eigh(matrix)
    eig_vectors = eig.eigenvectors[:, -n:]
    if mps:
        eig_vectors = eig_vectors.to("mps")
    return eig_vectors
