from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Optional
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics

from iceberg import Renderer, Colors

from torch.utils.data import DataLoader, Dataset

import jax
import jax.numpy as jnp

from abstractions import data, utils
from abstractions.abstraction import Model


class AnomalyDetector(ABC):
    def __init__(self, model: Model, params, max_batch_size: int = 4096):
        self.model = model
        self.params = params
        self.max_batch_size = max_batch_size

        self.forward_fn = jax.jit(
            lambda x: model.apply({"params": params}, x, return_activations=True)
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
    def train(self, dataset):
        """Train the anomaly detector with the given dataset as "normal" data."""

    def eval(
        self,
        normal_dataset: Dataset,
        anomalous_datasets: dict[str, Dataset],
        save_path: str | Path = "",
        histogram_percentile: int = 95,
    ):
        save_path = Path(save_path)

        normal_loader = DataLoader(
            normal_dataset,
            batch_size=self.max_batch_size,
            shuffle=False,
            collate_fn=data.numpy_collate,
        )
        anomalous_loaders = {
            k: DataLoader(
                ds,
                batch_size=self.max_batch_size,
                shuffle=False,
                collate_fn=data.numpy_collate,
            )
            for k, ds in anomalous_datasets.items()
        }

        normal_scores = []
        for batch in normal_loader:
            normal_scores.append(self.scores(batch))
        normal_scores = jnp.concatenate(normal_scores)

        anomalous_scores = {}
        metrics = {"AUC_ROC": {}}
        x_lim = jnp.percentile(normal_scores, histogram_percentile).item()
        for k, loader in anomalous_loaders.items():
            scores = []
            for batch in loader:
                scores.append(self.scores(batch))
            scores = jnp.concatenate(scores)
            anomalous_scores[k] = scores

            true_labels = jnp.concatenate(
                [jnp.ones_like(scores), jnp.zeros_like(normal_scores)]
            )
            all_scores = jnp.concatenate([scores, normal_scores])
            auc_roc = sklearn.metrics.roc_auc_score(
                y_true=true_labels,
                y_score=all_scores,
            )
            logger.log("METRICS", f"AUC_ROC ({k}): {auc_roc:.4f}")
            metrics["AUC_ROC"][k] = auc_roc

            # We use the most anomalous scores to compute the cutoff, to make sure
            # all score distributions are visible in the histogram
            x_lim = max(x_lim, jnp.percentile(scores, histogram_percentile).item())

        with open(save_path / "eval.json", "w") as f:
            json.dump(metrics, f)

        # Visualizations for anomaly scores
        plt.hist(
            normal_scores,
            bins=100,
            range=(normal_scores.min().item(), x_lim),
            alpha=0.5,
            label="Normal",
        )
        for k, scores in anomalous_scores.items():
            plt.hist(
                anomalous_scores,
                bins=100,
                range=(scores.min().item(), x_lim),
                alpha=0.5,
                label=k,
            )
        plt.legend()
        plt.xlabel("Anomaly score")
        plt.ylabel("Frequency")
        plt.title("Anomaly score distribution")
        plt.savefig(save_path / "histogram.pdf")

        # For now, we just plot the first anomalous dataset in the architecture figure,
        # not sure what I want to do here long term
        anomalous_dataset = anomalous_datasets[next(iter(anomalous_datasets.keys()))]
        layer_scores = self.layer_anomalies(anomalous_dataset)

        sample_loader = DataLoader(
            anomalous_dataset,
            batch_size=9,
            shuffle=False,
            collate_fn=data.numpy_collate,
        )
        sample_inputs = next(iter(sample_loader))
        if isinstance(sample_inputs, (tuple, list)):
            sample_inputs = sample_inputs[0]
        self.plot(layer_scores, path=save_path, inputs=sample_inputs)

    def _get_drawable(self, layer_scores, inputs):
        return self.model.get_drawable(layer_scores=layer_scores, inputs=inputs)

    def plot(
        self,
        layer_scores: Optional[jax.Array] = None,
        path: str | Path = "",
        inputs: Optional[np.ndarray] = None,
    ):
        plot = self._get_drawable(layer_scores, inputs)
        plot = plot.pad(10).scale(2)

        renderer = Renderer()
        renderer.render(plot, background_color=Colors.WHITE)
        path = Path(path)
        renderer.save_rendered_image(path / "architecture.png")

    def layer_anomalies(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.max_batch_size,
            shuffle=False,
            collate_fn=data.numpy_collate,
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

    def _get_trained_variables(self):
        return {}

    def _set_trained_variables(self, variables):
        pass

    def save(self, path: str | Path):
        logger.info(f"Saving detector to {utils.original_relative_path(path)}")
        utils.save(self._get_trained_variables(), path)

    def load(self, path: str | Path):
        logger.info(f"Loading detector from {utils.original_relative_path(path)}")
        self._set_trained_variables(utils.load(path))
