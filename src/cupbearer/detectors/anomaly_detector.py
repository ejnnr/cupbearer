import json
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

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
    def __init__(self, layer_aggregation: str = "mean"):
        # For storing the original detector variables when finetuning
        self.layer_aggregation = layer_aggregation
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
        dataset: MixedData | None = None,
        test_loader: DataLoader | None = None,
        batch_size: int = 1024,
        histogram_percentile: float = 95,
        save_path: Path | str | None = None,
        num_bins: int = 100,
        pbar: bool = False,
        layerwise: bool = False,
        log_yaxis: bool = True,
    ):
        test_loader = self.build_test_loaders(dataset, test_loader, batch_size)
        assert 0 < histogram_percentile <= 100

        if pbar:
            test_loader = tqdm(test_loader, desc="Evaluating", leave=False)

        scores, labels = self.compute_eval_scores(test_loader, layerwise=layerwise)
        
        return self.plot_scores(
            scores, labels, histogram_percentile, num_bins, log_yaxis, save_path
        )
    
    def build_test_loaders(
        self, 
        dataset: MixedData | None, 
        dataloader: DataLoader | None, 
        batch_size: int
    ) -> DataLoader:
        if dataloader is None:
            assert isinstance(dataset, MixedData), type(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            assert dataset is None, "Either dataset or dataloader should be provided, not both."
            assert isinstance(dataloader.dataset, MixedData), type(dataloader.dataset)
        return dataloader

    def compute_eval_scores(self, test_loader: DataLoader, layerwise: bool = False):
        scores = defaultdict(list)
        labels = defaultdict(list)
        # It's important we don't use torch.inference_mode() here, since we want
        # to be able to override this in certain detectors using torch.enable_grad().
        with torch.no_grad():
            for batch in test_loader:
                inputs, new_labels = batch
                if layerwise:
                    new_scores = self.layerwise_scores(inputs)
                else:
                    new_scores = {"all": self.scores(inputs)}
                for layer, score in new_scores.items():
                    if isinstance(score, torch.Tensor):
                        score = score.cpu().numpy()
                    assert score.shape == new_labels.shape
                    scores[layer].append(score)
                    labels[layer].append(new_labels)
        return scores, labels
    
    def plot_scores(self, scores, labels, histogram_percentile, num_bins, log_yaxis, save_path):
        metrics = defaultdict(dict)
        scores = {layer: np.concatenate(scores[layer]) for layer in scores}
        labels = {layer: np.concatenate(labels[layer]) for layer in labels}

        figs = {}

        for layer in scores:
            auc_roc = sklearn.metrics.roc_auc_score(
                y_true=labels[layer],
                y_score=scores[layer],
            )
            ap = sklearn.metrics.average_precision_score(
                y_true=labels[layer],
                y_score=scores[layer],
            )
            logger.info(f"AUC_ROC ({layer}): {auc_roc:.4f}")
            logger.info(f"AP ({layer}): {ap:.4f}")
            metrics[layer]["AUC_ROC"] = auc_roc
            metrics[layer]["AP"] = ap

            upper_lim = np.percentile(scores[layer], histogram_percentile).item()
            # Usually there aren't extremely low outliers, so we just use the minimum,
            # otherwise this tends to weirdly cut of the histogram.
            lower_lim = scores[layer].min().item()

            bins = np.linspace(lower_lim, upper_lim, num_bins)

            # Visualizations for anomaly scores
            fig, ax = plt.subplots()
            for i, name in enumerate(["Normal", "Anomalous"]):
                vals = scores[layer][labels[layer] == i]
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

        if not save_path:
            return metrics, figs

        save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        # Everything from here is just saving metrics and creating figures
        # (which we skip if they aren't going to be saved anyway).
        with open(save_path / "eval.json", "w") as f:
            json.dump(metrics, f)

        for layer, fig in figs.items():
            fig.savefig(save_path / f"histogram_{layer}.pdf")

        return metrics, figs

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
        if self.layer_aggregation == "mean":
            return sum(scores) / len(scores)  # type: ignore
        elif self.layer_aggregation == "max":
            return torch.amax(torch.stack(list(scores)), dim=0)
        else:
            raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")

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
