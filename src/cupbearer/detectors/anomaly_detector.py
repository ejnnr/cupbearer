import json
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sklearn.metrics
import torch
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cupbearer import utils
from cupbearer.data import MixedData
from cupbearer.tasks import Task

from .extractors import FeatureExtractor


class AnomalyDetector(ABC):
    """Base class for model-based anomaly detectors.

    These are the main detectors that users will interact with directly.

    Args:
        feature_extractor: A feature extractor to use. If None, the detector will pass
            only the raw input to subclasses.
        layer_aggregation: How to aggregate anomaly scores from different layers.
            Must be "mean" or "max".
        model: Usually doesn't need to be passed in here, see `set_model()`.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        layer_aggregation: str = "mean",
        model: torch.nn.Module | None = None,
    ):
        self.feature_extractor = feature_extractor
        self.layer_aggregation = layer_aggregation
        self.set_model(model)

    @abstractmethod
    def _compute_layerwise_scores(
        self, inputs: Any, features: Any
    ) -> dict[str, torch.Tensor]:
        """Compute anomaly scores for the given inputs for each layer.

        Each element of the returned dictionary must have shape (batch_size, ).

        If a detector can't compute layerwise scores, it should instead return
        a dictionary with only one element (by convention using an 'all' key).
        """

    @abstractmethod
    def _train(
        self,
        trusted_dataloader: DataLoader | None,
        untrusted_dataloader: DataLoader | None,
        **kwargs,
    ):
        """Train the anomaly detector with the given datasets on the given model.

        At least one of trusted_dataloader or untrusted_dataloader will be provided.

        The dataloaders return tuples (batch, features), where `batch` will be created
        directly from the underlying dataset (so potentially include labels) and
        `features` is None or the output of the feature extractor.
        """

    def _get_trained_variables(self):
        return {}

    def _set_trained_variables(self, variables):
        pass

    def set_model(self, model: torch.nn.Module | None):
        """Set the model used by the detector.

        In most cases, you don't need to call this method directly. The recommended
        workflow is simply:
        ```
        detector = MyDetector(...)
        detector.train(task)
        detector.eval(task)
        ```
        where `task` will contain both the model and the datasets. You can also specify
        a model at init (`MyDetector(model=model)`) or manually pass one to `train()`
        or `eval()`.

        Anomaly detectors "remember" their model, so if you ever pass one to any of
        `__init__()`, `train()`, `eval()`, or `set_model()`, it will be used for all
        subsequent calls to detector methods (until overriden). This includes models
        passed implicitly via a task.
        """
        # This is separate from __init__ because we want to be able to set the model
        # automatically based on the task, instead of letting the user pass it in.
        # On the other hand, it's separate from train() because we might need to set
        # the model even when just using the detector for inference.
        #
        # Subclasses can implement more complex logic here.
        self.model = model
        if self.feature_extractor:
            self.feature_extractor.set_model(model)

    def compute_layerwise_scores(self, inputs) -> dict[str, torch.Tensor]:
        """Compute anomaly scores for the given inputs for each layer.

        Args:
            inputs: a batch of input data to the model

        Returns:
            A dictionary with anomaly scores, each element has shape (batch_size, ).
        """
        if self.feature_extractor:
            features = self.feature_extractor(inputs)
        else:
            features = None
        return self._compute_layerwise_scores(inputs=inputs, features=features)

    def compute_scores(self, inputs) -> torch.Tensor:
        """Compute anomaly scores for the given inputs.

        Args:
            inputs: a batch of input data to the model

        Returns:
            Anomaly scores for the given inputs, of shape (batch_size, )
        """
        scores = self.compute_layerwise_scores(inputs)
        return self._aggregate_scores(scores)

    def _aggregate_scores(
        self, layerwise_scores: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        scores = layerwise_scores.values()
        assert len(scores) > 0
        # Type checker doesn't take into account that scores is non-empty,
        # so thinks this might be a float.
        if self.layer_aggregation == "mean":
            return sum(scores) / len(scores)  # type: ignore
        elif self.layer_aggregation == "max":
            return torch.amax(torch.stack(list(scores)), dim=0)
        else:
            raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")

    def _collate_fn(self, batch):
        batch = torch.utils.data.default_collate(batch)
        inputs = utils.inputs_from_batch(batch)
        if self.feature_extractor:
            features = self.feature_extractor(inputs)
        else:
            features = None
        return batch, features

    def train(
        self,
        task: Task | None = None,
        *,
        trusted_data: Dataset | None = None,
        untrusted_data: Dataset | None = None,
        model: torch.nn.Module | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ):
        """Train the anomaly detector with the given datasets on the given model.

        The recommended way to call this method is `detector.train(task)`, but it's also
        possible to manually pass datasets and a model instead of a task. The model does
        not need to be passed if one was specified earlier (e.g. during initialization
        or with `set_model()`).
        """
        if task is None:
            if trusted_data is None and untrusted_data is None:
                raise ValueError(
                    "Either task or trusted_data or untrusted_data must be provided."
                )
            if model is not None:
                self.set_model(model)
        else:
            assert model is None, "model must be None when passing a task"
            assert (
                trusted_data is None and untrusted_data is None
            ), "trusted_data and untrusted_data must be None when passing a task"
            trusted_data = task.trusted_data
            untrusted_data = task.untrusted_train_data
            self.set_model(task.model)

        dataloaders = []
        for data in [trusted_data, untrusted_data]:
            if data is None:
                dataloaders.append(None)
            else:
                dataloaders.append(
                    torch.utils.data.DataLoader(
                        data,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=self._collate_fn,
                    )
                )

        return self._train(
            trusted_dataloader=dataloaders[0],
            untrusted_dataloader=dataloaders[1],
            **kwargs,
        )

    def eval(
        self,
        task: Task | None = None,
        *,
        dataset: MixedData | None = None,
        test_loader: DataLoader | None = None,
        model: torch.nn.Module | None = None,
        batch_size: int = 1024,
        histogram_percentile: float = 95,
        save_path: Path | str | None = None,
        num_bins: int = 100,
        pbar: bool = False,
        layerwise: bool = False,
        log_yaxis: bool = True,
        show_worst_mistakes: bool = False,
        sample_format_fn: Callable[[Any], Any] | None = None,
    ):
        """Evaluate the anomaly detector on the given dataset.

        The recommended way to call this method is `detector.eval(task)`, but it's also
        possible to manually pass a dataset/dataloader and a model instead of a task.
        The model does not need to be passed if one was specified earlier (e.g. during
        initialization, training, or with `set_model()`).
        """
        if task is None:
            if model is not None:
                self.set_model(model)
        else:
            assert model is None, "model must be None when passing a task"
            assert (
                dataset is None and test_loader is None
            ), "dataset and test_loader must be None when passing a task"
            dataset = task.test_data
            self.set_model(task.model)

        test_loader = self.build_test_loaders(dataset, test_loader, batch_size)
        assert 0 < histogram_percentile <= 100

        if pbar:
            test_loader = tqdm(test_loader, desc="Evaluating", leave=False)

        scores, labels = self.compute_eval_scores(test_loader, layerwise=layerwise)

        return self.get_eval_results(
            scores,
            labels,
            histogram_percentile,
            num_bins,
            log_yaxis,
            save_path,
            show_worst_mistakes=show_worst_mistakes,
            sample_format_fn=sample_format_fn,
            dataset=dataset or test_loader.dataset,
        )

    def build_test_loaders(
        self, dataset: MixedData | None, dataloader: DataLoader | None, batch_size: int
    ) -> DataLoader:
        if dataloader is None:
            assert isinstance(dataset, MixedData), type(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            assert (
                dataset is None
            ), "Either dataset or dataloader should be provided, not both."
            assert isinstance(dataloader.dataset, MixedData), type(dataloader.dataset)
        return dataloader

    def compute_eval_scores(
        self, test_loader: DataLoader, layerwise: bool = False
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        scores = defaultdict(list)
        labels = []

        # It's important we don't use torch.inference_mode() here, since we want
        # to be able to override this in certain detectors using torch.enable_grad().
        with torch.no_grad():
            for batch in test_loader:
                samples, new_labels = batch
                inputs = utils.inputs_from_batch(samples)
                if layerwise:
                    new_scores = self.compute_layerwise_scores(inputs)
                    new_scores["all"] = self._aggregate_scores(new_scores)
                else:
                    # For some detectors, this is what we'd get anyway when calling
                    # compute_layerwise_scores, but for layerwise detectors, we want to
                    # make sure to analyze only the overall scores unless requested
                    # otherwise.
                    new_scores = {"all": self.compute_scores(inputs)}
                for layer, score in new_scores.items():
                    if isinstance(score, torch.Tensor):
                        score = score.cpu().numpy()
                    assert score.shape == new_labels.shape
                    scores[layer].append(score)
                labels.append(new_labels)
        scores = {layer: np.concatenate(scores[layer]) for layer in scores}
        labels = np.concatenate(labels)
        return scores, labels

    def get_eval_results(
        self,
        scores: dict[str, np.ndarray],
        labels: np.ndarray,
        histogram_percentile: float,
        num_bins: int,
        log_yaxis: bool,
        save_path: Path | str | None,
        show_worst_mistakes: bool = False,
        sample_format_fn: Callable[[Any], Any] | None = None,
        dataset: MixedData | None = None,
    ) -> tuple[dict[str, dict], dict[str, Figure]]:
        metrics = defaultdict(dict)

        figs = {}

        for layer in scores:
            auc_roc = sklearn.metrics.roc_auc_score(
                y_true=labels,
                y_score=scores[layer],
            ).item()
            ap = sklearn.metrics.average_precision_score(
                y_true=labels,
                y_score=scores[layer],
            ).item()
            logger.info(f"AUC_ROC ({layer}): {auc_roc:.4f}")
            logger.info(f"AP ({layer}): {ap:.4f}")
            metrics[layer]["AUC_ROC"] = auc_roc
            metrics[layer]["AP"] = ap
            
            # Save the scores for the positive and negative examples in metrics
            metrics[layer]["scores"] = {
                "positive": scores[layer][labels == 1].tolist(),
                "negative": scores[layer][labels == 0].tolist(),
            }

            upper_lim = np.percentile(scores[layer], histogram_percentile).item()
            # Usually there aren't extremely low outliers, so we just use the minimum,
            # otherwise this tends to weirdly cut of the histogram.
            lower_lim = scores[layer].min().item()

            bins = np.linspace(lower_lim, upper_lim, num_bins)

            # Visualizations for anomaly scores
            fig, ax = plt.subplots()
            for i, name in enumerate(["Normal", "Anomalous"]):
                vals = scores[layer][labels == i]
                ax.hist(
                    vals,
                    bins=bins,
                    alpha=0.5,
                    label=name,
                    log=log_yaxis,
                )
            ax.legend()
            ax.set_xlabel("Anomaly score")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Anomaly score distribution ({layer})")
            textstr = f"AUROC: {auc_roc:.1%}\n AP: {ap:.1%}"
            props = dict(boxstyle="round", facecolor="white")
            ax.text(
                0.98,
                0.80,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=props,
            )
            figs[layer] = fig

        if show_worst_mistakes and dataset is None:
            warnings.warn(
                "show_worst_mistakes=True requires a dataset to be provided but "
                "none was provided. Skipping worst mistakes."
            )
        elif show_worst_mistakes:
            assert isinstance(dataset, MixedData), type(dataset)
            for layer, layer_scores in scores.items():
                # "false positives" etc. isn't quite right because there's no threshold
                false_positives = np.argsort(
                    np.where(labels == 0, layer_scores, -np.inf)
                )[-10:]
                false_negatives = np.argsort(
                    np.where(labels == 1, layer_scores, np.inf)
                )[:10]

                print("\nNormal but high anomaly score:\n")
                for idx in false_positives:
                    sample, anomaly_label = dataset[idx]
                    assert anomaly_label == 0
                    if sample_format_fn:
                        sample = sample_format_fn(sample)
                    print(f"#{idx} ({layer_scores[idx]}): {sample}")
                print("\n====================================")
                print("Anomalous but low anomaly score:\n")
                for idx in false_negatives:
                    sample, anomaly_label = dataset[idx]
                    assert anomaly_label == 1
                    if sample_format_fn:
                        sample = sample_format_fn(sample)
                    print(f"#{idx} ({layer_scores[idx]}): {sample}")

        if not save_path:
            return metrics, figs

        save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "eval.json", "w") as f:
            json.dump(metrics, f)

        for layer, fig in figs.items():
            fig.savefig(save_path / f"histogram_{layer}.pdf")

        return metrics, figs

    def save_weights(self, path: str | Path, overwrite: bool = False):
        logger.info(f"Saving detector to {path}")
        utils.save(self._get_trained_variables(), path, overwrite=overwrite)

    def load_weights(self, path: str | Path):
        logger.info(f"Loading detector from {path}")
        self._set_trained_variables(utils.load(path))
