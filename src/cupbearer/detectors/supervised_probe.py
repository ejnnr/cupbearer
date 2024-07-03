from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import tqdm

from cupbearer.data import MixedData
from cupbearer.detectors.activation_based import ActivationBasedDetector


class SupervisedLinearProbe(ActivationBasedDetector):

    def __init__(self, scaler = StandardScaler | None , **kwargs):
        self.scaler = scaler
        super().__init__(**kwargs)

    def train(
        self,
        trusted_data,
        untrusted_data,
        save_path: Path | str,
        *,
        batch_size: int = 64,
        **sklearn_kwargs,
    ):
        if untrusted_data is None:
            raise ValueError("Supervised probe requires untrusted training data.")

        assert isinstance(untrusted_data, MixedData)
        if not untrusted_data.return_anomaly_labels:
            raise ValueError(
                "The supervised probe is a skyline detector meant to be trained "
                "with access to anomaly labels."
            )

        dataloader = DataLoader(untrusted_data, batch_size=batch_size, shuffle=True)

        activations = []
        anomaly_labels = []
        for batch in tqdm.tqdm(dataloader):
            rest, new_anomaly_labels = batch
            new_activations = self.get_activations(rest)
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

    def layerwise_scores(self, batch):
        activations = self.get_activations(batch)
        if len(activations) > 1:
            raise NotImplementedError(
                "The supervised probe only supports a single layer right now."
            )
        transform = self.scaler.transform if self.scaler is not None else lambda x: x
        activations = next(iter(activations.values()))
        return {
            # Get probabilities of class 1 (anomalous)
            next(iter(activations)): self.clf.predict_proba(
                transform(activations.cpu().numpy())
            )[:, 1]
        }

    def _get_trained_variables(self, saving: bool = False):
        # This will just pickle things, which maybe isn't the most efficient but easiest
        return self.clf

    def _set_trained_variables(self, variables):
        self.clf = variables
