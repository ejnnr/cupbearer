from abc import ABC, abstractmethod

import flax.linen as nn
import jax


class AnomalyDetector(ABC):
    def __init__(self, model: nn.Module, params):
        self.model = model
        self.params = params

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

    def train(self, dataset):
        """Train the anomaly detector with the given dataset as "normal" data."""
        self.trained = True
        return self._train(dataset)

    def scores(self, batch):
        """Compute anomaly scores for the given inputs.

        Args:
            inputs: a batch of input data to the model (potentially including labels).

        Returns:
            A batch of anomaly scores for the inputs.
        """
        if not self.trained:
            raise RuntimeError("Anomaly detector must be trained first.")
        return self._scores(batch)

    @abstractmethod
    def _train(self, dataset):
        pass

    @abstractmethod
    def _scores(self, batch):
        pass
