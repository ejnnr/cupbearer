from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from loguru import logger
from matplotlib import pyplot as plt
import sklearn.metrics

from iceberg import Bounds, Renderer, Colors
from iceberg.primitives import Blank

from torch.utils.data import DataLoader

import flax.linen as nn
import jax
import jax.numpy as jnp

from abstractions import data


class AnomalyDetector(ABC):
    def __init__(self, model: nn.Module, params, max_batch_size: int = 4096):
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

    def train(self, *args, **kwargs):
        """Train the anomaly detector with the given dataset as "normal" data."""
        self.trained = True
        return self._train(*args, **kwargs)

    def scores(self, batch, layerwise=False):
        """Compute anomaly scores for the given inputs.

        Args:
            batch: a batch of input data to the model (potentially including labels).
            layerwise: if True, return a list of scores for each layer of the model.

        Returns:
            A batch of anomaly scores for the inputs.
        """
        if not self.trained:
            raise RuntimeError("Anomaly detector must be trained first.")
        if layerwise:
            return self._layerwise_scores(batch)
        return self._scores(batch)

    def eval(self, normal_dataset, anomalous_dataset, save_path: str | Path = ""):
        normal_loader = DataLoader(
            normal_dataset,
            batch_size=self.max_batch_size,
            shuffle=False,
            collate_fn=data.numpy_collate,
        )
        anomalous_loader = DataLoader(
            anomalous_dataset,
            batch_size=self.max_batch_size,
            shuffle=False,
            collate_fn=data.numpy_collate,
        )

        normal_scores = []
        for batch in normal_loader:
            normal_scores.append(self.scores(batch))
        normal_scores = jnp.concatenate(normal_scores)

        anomalous_scores = []
        for batch in anomalous_loader:
            anomalous_scores.append(self.scores(batch))
        anomalous_scores = jnp.concatenate(anomalous_scores)

        true_labels = jnp.concatenate(
            [jnp.ones_like(anomalous_scores), jnp.zeros_like(normal_scores)]
        )
        all_scores = jnp.concatenate([anomalous_scores, normal_scores])
        auc_roc = sklearn.metrics.roc_auc_score(
            y_true=true_labels,
            y_score=all_scores,
        )
        logger.log("METRICS", f"AUC_ROC: {auc_roc:.4f}")

        x_lim = jnp.percentile(all_scores, 95)

        # Visualizations for consistency losses
        plt.hist(
            normal_scores,
            bins=100,
            range=(normal_scores.min(), min(normal_scores.max().item(), x_lim)),
            alpha=0.5,
            label="Normal",
        )
        plt.hist(
            anomalous_scores,
            bins=100,
            range=(anomalous_scores.min(), min(normal_scores.max().item(), x_lim)),
            alpha=0.5,
            label="Anomalous",
        )
        plt.legend()
        plt.xlabel("Anomaly score")
        plt.ylabel("Frequency")
        plt.title("Anomaly score distribution")
        plt.savefig("histogram.pdf")

        layer_scores = self.layer_anomalies(anomalous_dataset)

        self.plot(layer_scores, path=save_path)

    def _get_drawable(self, layer_scores):
        return self.model.get_drawable(layer_scores)

    def plot(self, layer_scores: Optional[jax.Array] = None, path: str | Path = ""):
        drawing = self._get_drawable(layer_scores)
        canvas = Blank(drawing.bounds.inset(-10), Colors.WHITE)
        scene = canvas.add_centered(drawing)
        scene = scene.scale(2)

        renderer = Renderer()
        renderer.render(scene)
        if not isinstance(path, Path):
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
            new_scores = self.scores(batch, layerwise=True)
            # Sum over batch axis
            scores = scores + new_scores.sum(axis=1)
            num_elements += new_scores.shape[1]
        # We're also taking the mean over the dataset
        scores = scores / num_elements
        return scores

    @abstractmethod
    def _train(self, dataset):
        pass

    @abstractmethod
    def _layerwise_scores(self, batch) -> jax.Array:
        pass

    def _scores(self, batch):
        return self._layerwise_scores(batch).mean(axis=0)
