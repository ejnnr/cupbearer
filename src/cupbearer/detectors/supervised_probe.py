import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cupbearer import utils
from cupbearer.data import MixedData
from cupbearer.detectors.activation_based import ActivationBasedDetector


class SupervisedLinearProbe(ActivationBasedDetector):
    def __init__(self, scaler: StandardScaler | None = None, **kwargs):
        self.scaler = scaler
        super().__init__(**kwargs)

    def _collate_fn(self, batch):
        batch = torch.utils.data.default_collate(batch)
        # Needs to be overriden becaues we need to remove anomaly labels before passing
        # this to the model. So we just add this extra line:
        batch, anomaly_labels = batch
        inputs = utils.inputs_from_batch(batch)
        if self.feature_extractor:
            features = self.feature_extractor(inputs)
        else:
            features = None
        return anomaly_labels, features

    def _train(
        self,
        trusted_dataloader,
        untrusted_dataloader,
        **sklearn_kwargs,
    ):
        if untrusted_dataloader is None:
            raise ValueError("Supervised probe requires untrusted training data.")

        assert isinstance(untrusted_dataloader.dataset, MixedData)
        if not untrusted_dataloader.dataset.return_anomaly_labels:
            raise ValueError(
                "The supervised probe is a skyline detector meant to be trained "
                "with access to anomaly labels."
            )

        activations = []
        anomaly_labels = []
        for batch in tqdm.tqdm(untrusted_dataloader):
            # See the custom _collate_fn earlier; it directly returns anomaly labels
            # instead of the full inputs.
            new_anomaly_labels, new_activations = batch
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

    def _compute_layerwise_scores(self, inputs, features):
        if len(features) > 1:
            raise NotImplementedError(
                "The supervised probe only supports a single layer right now."
            )
        transform = self.scaler.transform if self.scaler is not None else lambda x: x
        name, features = next(iter(features.items()))
        return {
            # Get probabilities of class 1 (anomalous)
            name: self.clf.predict_proba(transform(features.cpu().numpy()))[:, 1]
        }

    def _get_trained_variables(self):
        # This will just pickle things, which maybe isn't the most efficient but easiest
        return self.clf

    def _set_trained_variables(self, variables):
        self.clf = variables
