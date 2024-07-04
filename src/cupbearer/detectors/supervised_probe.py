from pathlib import Path
from typing import Any, Callable

import torch
import tqdm
from sklearn.linear_model import LogisticRegression

from .anomaly_detector import LayerwiseAnomalyDetector
from .extractors import ActivationExtractor, FeatureExtractor


class SupervisedLinearProbe(LayerwiseAnomalyDetector):
    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        activation_names: list[str] | None = None,
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        global_processing_fn: Callable[
            [dict[str, torch.Tensor]], dict[str, torch.Tensor]
        ]
        | None = None,
    ):
        if feature_extractor is None and activation_names is None:
            raise ValueError(
                "Either a feature extractor or a list of activation names "
                "must be provided."
            )
        super().__init__(
            feature_extractor=feature_extractor,
            default_extractor_kwargs={
                "names": activation_names,
                "individual_processing_fn": individual_processing_fn,
                "global_processing_fn": global_processing_fn,
            },
        )

    def _default_extractor_factory(self, **kwargs):
        return ActivationExtractor(return_inputs=True, **kwargs)

    def _train(
        self,
        trusted_dataloader,
        untrusted_dataloader,
        save_path: Path | str,
        *,
        batch_size: int = 64,
        **sklearn_kwargs,
    ):
        if untrusted_dataloader is None:
            raise ValueError("Supervised probe requires untrusted training data.")

        # assert isinstance(untrusted_data, MixedData)
        # if not untrusted_data.return_anomaly_labels:
        #     raise ValueError(
        #         "The supervised probe is a skyline detector meant to be trained "
        #         "with access to anomaly labels."
        #     )

        activations = []
        anomaly_labels = []
        for batch in tqdm.tqdm(untrusted_dataloader):
            # TODO: inputs are only the actual inputs, so this won't work
            (rest, new_anomaly_labels) = batch.pop("inputs")
            new_activations = batch
            if len(new_activations) > 1:
                raise NotImplementedError(
                    "The supervised probe only supports a single layer right now."
                )
            new_activations = next(iter(new_activations.values()))
            activations.append(new_activations)
            anomaly_labels.append(new_anomaly_labels)

        activations = torch.cat(activations)
        anomaly_labels = torch.cat(anomaly_labels)

        self.clf = LogisticRegression(**sklearn_kwargs)
        self.clf.fit(activations.cpu().numpy(), anomaly_labels.cpu().numpy())

    def _compute_layerwise_scores(self, batch):
        batch.pop("inputs")
        activations = batch
        if len(activations) > 1:
            raise NotImplementedError(
                "The supervised probe only supports a single layer right now."
            )
        name, activations = next(iter(activations.items()))
        return {
            # Get probabilities of class 1 (anomalous)
            name: self.clf.predict_proba(activations.cpu().numpy())[:, 1]
        }

    def _get_trained_variables(self, saving: bool = False):
        # This will just pickle things, which maybe isn't the most efficient but easiest
        return self.clf

    def _set_trained_variables(self, variables):
        self.clf = variables
