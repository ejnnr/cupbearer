import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import sklearn.metrics
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cupbearer.data import TestDataMix, numpy_collate
from cupbearer.models.computations import Model
from cupbearer.utils import utils


class AnomalyDetector(ABC):
    def __init__(
        self,
        model: Model,
        params,
        rng,
        max_batch_size: int = 4096,
        save_path: Path | str | None = None,
    ):
        self.model = model
        self.params = params
        self.rng = rng
        # For storing the original detector variables when finetuning
        self._original_variables = None
        self.max_batch_size = max_batch_size
        self.save_path = None if save_path is None else Path(save_path)

        self.forward_fn = jax.jit(
            lambda x: self.model.apply(
                {"params": self.params}, x, return_activations=True
            )
        )

        self.trained = False

    def _model(self, batch):
        # batch may contain labels or other info, if so we strip it out
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch
        output, activations = self.forward_fn(inputs)
        return output, activations

    @abstractmethod
    def train(self, dataset, debug: bool = False, **kwargs):
        """Train the anomaly detector with the given dataset as "normal" data."""

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
        train_dataset: Dataset,
        test_dataset: TestDataMix,
        histogram_percentile: float = 95,
        num_bins: int = 100,
        pbar: bool = False,
    ):
        # Check this explicitly because otherwise things can break in weird ways
        # when we assume that anomaly labels are included.
        assert isinstance(test_dataset, TestDataMix), type(test_dataset)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.max_batch_size,
            # For some methods, such as adversarial abstractions, it might matter how
            # normal/anomalous data is distributed into batches. In that case, we want
            # to mix them by default.
            shuffle=True,
            collate_fn=numpy_collate,
        )

        metrics = {}
        assert 0 < histogram_percentile <= 100

        scores = []
        layer_scores = 0
        num_elements = 0
        # Normal=0, Anomalous=1
        labels = []
        if pbar:
            test_loader = tqdm(test_loader, desc="Evaluating", leave=False)
        for batch in test_loader:
            inputs, new_labels = batch
            new_scores = self.layerwise_scores(inputs)
            scores.append(new_scores.mean(axis=0))
            # Sum over batch axis
            layer_scores = layer_scores + new_scores.sum(axis=1)
            num_elements += new_scores.shape[1]
            labels.append(new_labels)
        assert isinstance(layer_scores, jax.Array)
        # We're also taking the mean over the dataset:
        layer_scores = layer_scores / num_elements
        scores = jnp.concatenate(scores)
        labels = jnp.concatenate(labels)

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

        upper_lim = jnp.percentile(scores, histogram_percentile).item()
        # Usually there aren't extremely low outliers, so we just use the minimum,
        # otherwise this tends to weirdly cut of the histogram.
        lower_lim = scores.min().item()

        bins = np.linspace(lower_lim, upper_lim, num_bins)

        if not self.save_path:
            return

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

        sample_loader = DataLoader(
            test_dataset,
            batch_size=9,
            # Shuffling to ideally get a mix of normal and anomalous data
            shuffle=True,
            collate_fn=numpy_collate,
        )
        sample_inputs, _ = next(iter(sample_loader))
        if isinstance(sample_inputs, (tuple, list)):
            sample_inputs = sample_inputs[0]

        self.plot(layer_scores, inputs=sample_inputs)

    def _get_drawable(self, layer_scores, inputs):
        return self.model.get_drawable(layer_scores=layer_scores, inputs=inputs)

    def plot(
        self,
        layer_scores: Optional[jax.Array] = None,
        inputs: Optional[np.ndarray] = None,
    ):
        try:
            from iceberg import Colors, Renderer
        except ImportError:
            logger.info(
                "Skipping architecture visualization because iceberg is missing"
            )
            return

        if not self.save_path:
            raise ValueError("No save path set")

        plot = self._get_drawable(layer_scores, inputs)
        plot = plot.pad(10).scale(2)

        renderer = Renderer()
        renderer.render(plot, background_color=Colors.WHITE)
        renderer.save_rendered_image(self.save_path / "architecture.png")

    def layer_anomalies(self, dataset: Dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.max_batch_size,
            shuffle=False,
            collate_fn=numpy_collate,
        )
        scores = 0
        num_elements = 0
        for batch in dataloader:
            new_scores = self.layerwise_scores(batch)
            # Sum over batch axis
            scores = scores + new_scores.sum(axis=1)
            num_elements += new_scores.shape[1]
        # We're also taking the mean over the dataset
        scores = scores / num_elements
        return scores

    @abstractmethod
    def layerwise_scores(self, batch) -> jax.Array:
        """Compute anomaly scores for the given inputs for each layer.

        Args:
            batch: a batch of input data to the model (potentially including labels).

        Returns:
            An array of anomaly scores with shape (n_layers, batch).
        """

    def scores(self, batch):
        """Compute anomaly scores for the given inputs.

        Args:
            batch: a batch of input data to the model (potentially including labels).

        Returns:
            A batch of anomaly scores for the inputs.
        """
        return self.layerwise_scores(batch).mean(axis=0)

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
