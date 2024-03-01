import json
from abc import ABC, abstractmethod
from collections.abc import Collection
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sklearn.metrics
import torch
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cupbearer.data import MixedData
from cupbearer.models.models import HookedModel
from cupbearer.utils import utils


class AnomalyDetector(ABC):
    def __init__(
        self,
        max_batch_size: int = 4096,
        save_path: Optional[Path | str] = None,
    ):
        # For storing the original detector variables when finetuning
        self._original_variables = None
        self.max_batch_size = max_batch_size
        self.save_path = None if save_path is None else Path(save_path)

        self.trained = False

    def set_model(self, model: HookedModel):
        # This is separate from __init__ because we want to be able to set the model
        # automatically based on the task, instead of letting the user pass it in.
        # On the other hand, it's separate from train() because we might need to set
        # the model even when just using the detector for inference.
        #
        # Subclasses can implement more complex logic here.
        self.model = model
        self.trained = False

    @abstractmethod
    def train(
        self,
        trusted_data: Dataset | None,
        untrusted_data: Dataset | None,
        *,
        num_classes: int,
        train_config: utils.BaseConfig,
    ):
        """Train the anomaly detector with the given datasets on the given model.

        At least one of trusted_data or untrusted_data must be provided.
        """

    @contextmanager
    def finetune(self, **kwargs):
        """Tune the anomaly detector.

        The finetuned parameters will be stored in this detector alongside the original
        ones. Within the context manager block, the detector will use the finetuned
        parameters (e.g. when calling `eval`). At the end of the block, the finetuned
        parameters will be removed. To store the finetuned parameters permanently,
        you can access the value the context manager yields.

        Might not be available for all anomaly detectors.

        Example:
        ```
        with detector.finetune(normal_dataset, new_dataset) as finetuned_params:
            detector.eval(normal_dataset, new_dataset) # uses finetuned params
            scores = detector.scores(some_other_dataset)
            utils.save(finetuned_params, "finetuned_params")

        detector.eval(normal_dataset, new_dataset) # uses original params
        ```
        """
        self._original_vars = self._get_trained_variables()
        finetuned_vars = self._finetune(**kwargs)
        self._set_trained_variables(finetuned_vars)
        yield finetuned_vars
        if self._original_vars:
            # original_vars might be empty if the detector was never trained
            self._set_trained_variables(self._original_vars)
        self._original_vars = None

    def _finetune(self, **kwargs) -> dict:
        """Finetune the anomaly detector to try to flag the new data as anomalous.

        Should return variables for the detector that can be passed to
        `_set_trained_variables`.
        """
        raise NotImplementedError(
            f"Finetuning not implemented for {self.__class__.__name__}."
        )

    def eval(
        self,
        # Don't need train_dataset here, but e.g. adversarial abstractions need it,
        # and in general there's no reason to deny detectors access to it during eval.
        # TODO: I think we can/should remove this and require detectors to handle
        # anything involving training data during training (now that they get access
        # to untrusted data then).
        train_dataset: Dataset,
        test_dataset: MixedData,
        histogram_percentile: float = 95,
        num_bins: int = 100,
        pbar: bool = False,
    ):
        # Check this explicitly because otherwise things can break in weird ways
        # when we assume that anomaly labels are included.
        assert isinstance(test_dataset, MixedData), type(test_dataset)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.max_batch_size,
            # For some methods, such as adversarial abstractions, it might matter how
            # normal/anomalous data is distributed into batches. In that case, we want
            # to mix them by default.
            shuffle=True,
        )

        metrics = {}
        assert 0 < histogram_percentile <= 100

        scores = []
        # Normal=0, Anomalous=1
        labels = []
        if pbar:
            test_loader = tqdm(test_loader, desc="Evaluating", leave=False)
        with torch.inference_mode():
            for batch in test_loader:
                inputs, new_labels = batch
                scores.append(self.scores(inputs).cpu().numpy())
                labels.append(new_labels)
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)

        auc_roc = sklearn.metrics.roc_auc_score(
            y_true=labels,
            y_score=scores,
        )
        ap = sklearn.metrics.average_precision_score(
            y_true=labels,
            y_score=scores,
        )
        logger.info(f"AUC_ROC: {auc_roc:.4f}")
        logger.info(f"AP: {ap:.4f}")
        metrics["AUC_ROC"] = auc_roc
        metrics["AP"] = ap

        upper_lim = np.percentile(scores, histogram_percentile).item()
        # Usually there aren't extremely low outliers, so we just use the minimum,
        # otherwise this tends to weirdly cut of the histogram.
        lower_lim = scores.min().item()

        bins = np.linspace(lower_lim, upper_lim, num_bins)

        if not self.save_path:
            return

        self.save_path.mkdir(parents=True, exist_ok=True)

        # Everything from here is just saving metrics and creating figures
        # (which we skip if they aren't going to be saved anyway).
        with open(self.save_path / "eval.json", "w") as f:
            json.dump(metrics, f)

        # Visualizations for anomaly scores
        for i, name in enumerate(["Normal", "Anomalous"]):
            vals = scores[labels == i]
            plt.hist(
                vals,
                bins=bins,
                alpha=0.5,
                label=name,
            )
        plt.legend()
        plt.xlabel("Anomaly score")
        plt.ylabel("Frequency")
        plt.title("Anomaly score distribution")
        plt.savefig(self.save_path / "histogram.pdf")

    @abstractmethod
    def layerwise_scores(self, batch) -> dict[str, torch.Tensor]:
        """Compute anomaly scores for the given inputs for each layer.

        You can just raise a NotImplementedError here for detectors that don't compute
        layerwise scores. In that case, you need to override `scores`. For detectors
        that can compute layerwise scores, you should override this method instead
        of `scores` since allows some additional metrics to be computed.

        Args:
            batch: a batch of input data to the model (potentially including labels).

        Returns:
            A dictionary with anomaly scores, each element has shape (batch, ).
        """

    def scores(self, batch) -> torch.Tensor:
        """Compute anomaly scores for the given inputs.

        If you override this, then your implementation of `layerwise_scores()`
        needs to raise a NotImplementedError. Implementing both this and
        `layerwise_scores()` is not supported.

        Args:
            batch: a batch of input data to the model (potentially including labels).

        Returns:
            A batch of anomaly scores for the inputs.
        """
        scores = self.layerwise_scores(batch).values()
        assert len(scores) > 0
        # Type checker doesn't take into account that scores is non-empty,
        # so thinks this might be a float.
        return sum(v for v in scores) / len(scores)  # type: ignore

    def _get_trained_variables(self, saving: bool = False):
        return {}

    def _set_trained_variables(self, variables):
        pass

    def save_weights(self, path: str | Path):
        logger.info(f"Saving detector to {path}")
        utils.save(self._get_trained_variables(saving=True), path)

    def load_weights(self, path: str | Path):
        logger.info(f"Loading detector from {path}")
        self._set_trained_variables(utils.load(path))


def default_activation_name_func(model):
    return model.default_names


class ActivationBasedDetector(AnomalyDetector):
    """AnomalyDetector using activations."""

    def __init__(
        self,
        activation_name_func: str
        | Callable[[HookedModel], Collection[str]]
        | None = None,
        max_batch_size: int = 4096,
        save_path: Path | str | None = None,
    ):
        super().__init__(max_batch_size=max_batch_size, save_path=save_path)

        if activation_name_func is None:
            activation_name_func = default_activation_name_func
        elif isinstance(activation_name_func, str):
            activation_name_func = utils.get_object(activation_name_func)

        assert callable(activation_name_func)  # make type checker happy

        self.activation_name_func = activation_name_func

    def set_model(self, model: HookedModel):
        super().set_model(model)
        self.activation_names = self.activation_name_func(model)

    def get_activations(self, batch):
        inputs = utils.inputs_from_batch(batch)
        return self.model.get_activations(inputs, self.activation_names)
