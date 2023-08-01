import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import sklearn.metrics
from iceberg import Colors, Renderer
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

from abstractions.data import _shared
from abstractions.models.computations import Model
from abstractions.utils import utils


class AnomalyDetector(ABC):
    def __init__(
        self,
        model: Model,
        params,
        max_batch_size: int = 4096,
        save_path: Path | str | None = None,
    ):
        self.model = model
        self.params = params
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
        normal_dataset: Dataset,
        anomalous_datasets: dict[str, Dataset],
        histogram_percentile: float = 95,
        num_bins: int = 100,
        plot_all_hists: bool = False,
    ):
        normal_loader = DataLoader(
            normal_dataset,
            batch_size=self.max_batch_size,
            shuffle=False,
            collate_fn=_shared.numpy_collate,
        )
        anomalous_loaders = {
            k: DataLoader(
                ds,
                batch_size=self.max_batch_size,
                shuffle=False,
                collate_fn=_shared.numpy_collate,
            )
            for k, ds in anomalous_datasets.items()
        }

        normal_scores = []
        for batch in normal_loader:
            normal_scores.append(self.scores(batch))
        normal_scores = jnp.concatenate(normal_scores)

        anomalous_scores = {}
        metrics = {"AUC_ROC": {}, "AP": {}}
        assert 0 < histogram_percentile <= 100
        histogram_percentile += 0.5 * (100 - histogram_percentile)
        lower_lim = jnp.percentile(normal_scores, 100 - histogram_percentile).item()
        upper_lim = jnp.percentile(normal_scores, histogram_percentile).item()
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
            ap = sklearn.metrics.average_precision_score(
                y_true=true_labels,
                y_score=all_scores,
            )
            logger.info(f"AUC_ROC ({k}): {auc_roc:.4f}")
            logger.info(f"AP ({k}): {ap:.4f}")
            metrics["AUC_ROC"][k] = auc_roc
            metrics["AP"][k] = ap

            # We use the most anomalous scores to compute the cutoff, to make sure
            # all score distributions are visible in the histogram
            upper_lim = max(
                upper_lim, jnp.percentile(scores, histogram_percentile).item()
            )
            lower_lim = min(
                lower_lim, jnp.percentile(scores, 100 - histogram_percentile).item()
            )

        bins = np.linspace(lower_lim, upper_lim, num_bins)

        if not self.save_path:
            return

        # Everything from here is just saving metrics and creating figures
        # (which we skip if they aren't going to be saved anyway).
        with open(self.save_path / "eval.json", "w") as f:
            json.dump(metrics, f)

        # Visualizations for anomaly scores
        plt.hist(
            normal_scores,
            bins=bins,
            alpha=0.5,
            label="Normal",
        )
        if plot_all_hists:
            for k, scores in list(anomalous_scores.items()):
                plt.hist(
                    scores,
                    bins=bins,
                    alpha=0.5,
                    label=k,
                )
        else:
            k, scores = next(iter(anomalous_scores.items()))
            plt.hist(
                scores,
                bins=bins,
                alpha=0.5,
                label=k,
            )
        plt.legend()
        plt.xlabel("Anomaly score")
        plt.ylabel("Frequency")
        plt.title("Anomaly score distribution")
        plt.savefig(self.save_path / "histogram.pdf")

        # For now, we just plot the first anomalous dataset in the architecture figure,
        # not sure what I want to do here long term
        anomalous_dataset = anomalous_datasets[next(iter(anomalous_datasets.keys()))]
        layer_scores = self.layer_anomalies(anomalous_dataset)

        sample_loader = DataLoader(
            anomalous_dataset,
            batch_size=9,
            shuffle=False,
            collate_fn=_shared.numpy_collate,
        )
        sample_inputs = next(iter(sample_loader))
        if isinstance(sample_inputs, (tuple, list)):
            sample_inputs = sample_inputs[0]

        try:
            self.plot(layer_scores, inputs=sample_inputs)
        except RuntimeError as e:
            if str(e) == "glfw.init() failed":
                logger.warning("Skipping architecture plot")
            else:
                raise

    def _get_drawable(self, layer_scores, inputs):
        return self.model.get_drawable(layer_scores=layer_scores, inputs=inputs)

    def plot(
        self,
        layer_scores: Optional[jax.Array] = None,
        inputs: Optional[np.ndarray] = None,
    ):
        if not self.save_path:
            raise ValueError("No save path set")

        plot = self._get_drawable(layer_scores, inputs)
        plot = plot.pad(10).scale(2)

        renderer = Renderer()
        renderer.render(plot, background_color=Colors.WHITE)
        renderer.save_rendered_image(self.save_path / "architecture.png")

    def layer_anomalies(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.max_batch_size,
            shuffle=False,
            collate_fn=_shared.numpy_collate,
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

    def save_weights(self, path: str | Path):
        logger.info(f"Saving detector to {path}")
        utils.save(self._get_trained_variables(), path)

    def load_weights(self, path: str | Path):
        logger.info(f"Loading detector from {path}")
        self._set_trained_variables(utils.load(path))

    @property
    def init_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for creating a copy of this detector instance.

        Shouldn't include `model` or `params`.

        Child classes will usually need to override this to make saving work correctly.
        The easiest way to do that is to use the `utils.storable` decorator on the
        child class.
        """
        return {"max_batch_size": self.max_batch_size}

    def save(self):
        if not self.save_path:
            raise ValueError("No save path set")
        logger.info(f"Saving detector to {self.save_path}")
        variables = self._get_trained_variables()
        module = self.__class__.__module__
        class_name = self.__class__.__qualname__
        path = module + "." + class_name
        hparams = self.init_kwargs
        utils.save(
            {
                "path": path,
                "hparams": hparams,
                "variables": variables,
            },
            self.save_path / "detector",
        )

    @classmethod
    def load(cls, cfg: str | Path, model: Model, params):
        ckpt = utils.load(cfg)
        class_ = utils.get_object(ckpt["path"])
        detector = class_(model=model, params=params, **ckpt["hparams"])
        detector._set_trained_variables(ckpt["variables"])
        return detector
