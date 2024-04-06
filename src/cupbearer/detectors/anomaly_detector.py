import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sklearn.metrics
import torch
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cupbearer import utils
from cupbearer.data import MixedData


class AnomalyDetector(ABC):
    def __init__(self):
        # For storing the original detector variables when finetuning
        self._original_variables = None
        self.trained = False

    def set_model(self, model: torch.nn.Module):
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
        save_path: Path | str | None,
        **kwargs,
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
        dataset: MixedData,
        batch_size: int = 1024,
        histogram_percentile: float = 95,
        save_path: Path | str | None = None,
        num_bins: int = 100,
        pbar: bool = False,
    ):
        # Check this explicitly because otherwise things can break in weird ways
        # when we assume that anomaly labels are included.
        assert isinstance(dataset, MixedData), type(dataset)

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
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

        if not save_path:
            return

        save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        # Everything from here is just saving metrics and creating figures
        # (which we skip if they aren't going to be saved anyway).
        with open(save_path / "eval.json", "w") as f:
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
        plt.savefig(save_path / "histogram.pdf")

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


class ActivationBasedDetector(AnomalyDetector):
    """AnomalyDetector using activations.

    Args:
        activation_names: The names of the activations to use for anomaly detection.
        activation_processing_func: A function to process the activations before
            computing the anomaly scores. The function should take the activations,
            the input data, and the name of the activations as arguments and return
            the processed activations.
    """

    def __init__(
        self,
        activation_names: list[str],
        activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
    ):
        super().__init__()
        self.activation_names = activation_names
        self.activation_processing_func = activation_processing_func

    def get_activations(self, batch):
        inputs = utils.inputs_from_batch(batch)
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
