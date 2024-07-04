from pathlib import Path

import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cupbearer.detectors.activation_based import ActivationBasedDetector


class SupervisedLinearProbe(ActivationBasedDetector):
    def __init__(self, scaler: StandardScaler | None = None, **kwargs):
        self.scaler = scaler
        super().__init__(**kwargs)

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

        activations = activations.cpu().numpy()
        anomaly_labels = anomaly_labels.cpu().numpy()

        if self.scaler is not None:
            activations = self.scaler.fit_transform(activations)
        self.clf = LogisticRegression(**sklearn_kwargs)
        self.clf.fit(activations, anomaly_labels)

    def _compute_layerwise_scores(self, batch):
        activations = batch
        if len(activations) > 1:
            raise NotImplementedError(
                "The supervised probe only supports a single layer right now."
            )
        transform = self.scaler.transform if self.scaler is not None else lambda x: x
        name, activations = next(iter(activations.items()))
        return {
            # Get probabilities of class 1 (anomalous)
            name: self.clf.predict_proba(transform(activations.cpu().numpy()))[:, 1]
        }

    def _get_trained_variables(self, saving: bool = False):
        # This will just pickle things, which maybe isn't the most efficient but easiest
        return self.clf

    def _set_trained_variables(self, variables):
        self.clf = variables
