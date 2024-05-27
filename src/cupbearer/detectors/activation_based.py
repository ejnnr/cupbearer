from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import tqdm
from torch.utils.data import Dataset

from cupbearer import utils
from cupbearer.data import MixedData

from .anomaly_detector import AnomalyDetector


class ActivationCache:
    """Cache for activations to speed up using multiple anomaly detectors.

    The main use case for this is if the model is expensive to run and we want to try
    many different detectors that require similar activations.

    The cache stores a dict from (input, activation_name) to the activations.
    """

    def __init__(self):
        """Create an empty cache."""
        self.cache: dict[tuple[Any, str], torch.Tensor] = {}
        # Just for debugging purposes:
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self.cache)

    def __contains__(self, key):
        return key in self.cache

    def count_missing(self, dataset: Dataset, activation_names: list[str]):
        """Count how many inputs from `dataset` are missing from the cache.

        `activation_names` is a list of the names of the activations we need to be
        in the cache. An input counts as missing if *some* of the activations are
        missing for that input.
        """
        count = 0
        for sample in dataset:
            if isinstance(dataset, MixedData) and dataset.return_anomaly_labels:
                sample = sample[0]
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                sample = sample[0]
            if not all((sample, name) in self.cache for name in activation_names):
                count += 1
        return count

    def store(self, path: str | Path):
        utils.save(self.cache, path)

    @classmethod
    def load(cls, path: str | Path):
        cache = cls()
        cache.cache = utils.load(path)
        return cache

    def get_activations(
        self,
        inputs,
        activation_names: list[str],
        activation_func: Callable[[Any], dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Get activations for a batch of inputs, using the cache if possible.

        If any activations are missing from the cache, they are computed and added
        to the cache.

        Args:
            inputs: The inputs to get activations for.
            activation_names: The names of the activations to get.
            activation_func: Takes in `inputs` and returns a dictionary of activations.

        Returns:
            A dict from activation name to the activations.
        """
        # We want to handle cases where some but not all elements are in the cache.
        missing_indices = []
        results: dict[str, list[torch.Tensor | None]] = defaultdict(
            lambda: [None] * len(inputs)
        )

        for i, input in enumerate(inputs):
            # The keys into the cache contain the input and the name of the activation.
            keys = [(input, name) for name in activation_names]
            # In principle we could support the case where some but not all activations
            # for a given input are already in the cache. If the missing activations
            # are early in the model, this might save some time since we wouldn't
            # have to do the full forward pass. But that seems like a niche use case
            # and not worth the added complexity. So for now, we recompute all
            # activations on inputs where some activations are missing.
            if all(key in self.cache for key in keys):
                self.hits += 1
                for name in activation_names:
                    results[name][i] = self.cache[(input, name)]
            else:
                missing_indices.append(i)

        if not missing_indices:
            return {name: torch.stack(results[name]) for name in activation_names}

        # Select the missing input elements, but make sure to keep the type.
        # Input could be a list/tuple of strings (for language models)
        # or tensors for images. (Unclear whether supporting tensors is important,
        # language models are the main case where having a cache is useful.)
        if isinstance(inputs, torch.Tensor):
            inputs = inputs[missing_indices]
        elif isinstance(inputs, list):
            inputs = [inputs[i] for i in missing_indices]
        elif isinstance(inputs, tuple):
            inputs = tuple(inputs[i] for i in missing_indices)
        else:
            raise NotImplementedError(
                f"Unsupported input type: {type(inputs)} of {inputs}"
            )

        new_acts = activation_func(inputs)
        self.misses += len(inputs)

        # Fill in the missing activations
        for name, act in new_acts.items():
            for i, idx in enumerate(missing_indices):
                results[name][idx] = act[i]
                self.cache[(inputs[i], name)] = act[i]

        assert all(
            all(result is not None for result in results[name])
            for name in activation_names
        )

        return {name: torch.stack(results[name]) for name in activation_names}


class ActivationBasedDetector(AnomalyDetector):
    """AnomalyDetector using activations.

    Args:
        activation_names: The names of the activations to use for anomaly detection.
        activation_processing_func: A function to process the activations before
            computing the anomaly scores. The function should take the activations,
            the input data, and the name of the activations as arguments and return
            the processed activations.
        cache: An ActivationCache to use for caching activations. If None (default),
            activations aren't cached. The cache is meant to be shared between multiple
            detectors on the same model.
        layer_aggregation: How to aggregate anomaly scores over layers to get a single
            global anomaly score. Options are "mean" (default) or "max".
    """

    def __init__(
        self,
        activation_names: list[str],
        activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        cache: ActivationCache | None = None,
        layer_aggregation: str = "mean",
    ):
        super().__init__(layer_aggregation=layer_aggregation)
        self.activation_names = activation_names
        self.activation_processing_func = activation_processing_func
        self.cache = cache

    def _get_activations_no_cache(self, inputs) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        inputs = utils.inputs_to_device(inputs, device)
        acts = utils.get_activations(self.model, self.activation_names, inputs)

        # Can be used to for example select activations at specific token positions
        if self.activation_processing_func is not None:
            acts = {
                k: self.activation_processing_func(v, inputs, k)
                for k, v in acts.items()
            }

        return acts

    def get_activations(self, batch) -> dict[str, torch.Tensor]:
        inputs = utils.inputs_from_batch(batch)

        if self.cache is None:
            return self._get_activations_no_cache(inputs)

        return self.cache.get_activations(
            inputs, self.activation_names, self._get_activations_no_cache
        )


class CacheBuilder(ActivationBasedDetector):
    """A dummy detector meant only for creating activation caches.

    Scores are meaningless. You can also just create a cache manually and call
    get_activations() on it, the only advantage of this is that it implements the
    detector interface, so you can use it with the exact same dataloading setup as
    the real detector.
    """

    def __init__(
        self,
        cache_path: str | Path,
        activation_names: list[str],
        activation_processing_func: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        cache: ActivationCache | None = None,
    ):
        if cache is None:
            cache = ActivationCache()
        super().__init__(activation_names, activation_processing_func, cache=cache)
        self.cache_path = cache_path

    def store_cache(self):
        assert self.cache is not None
        self.cache.store(self.cache_path)

    def train(self, trusted_data, untrusted_data, save_path, *, batch_size: int = 64):
        for data in [trusted_data, untrusted_data]:
            if data is None:
                continue
            dataloader = torch.utils.data.DataLoader(
                data, batch_size=batch_size, shuffle=False
            )
            for batch in tqdm.tqdm(dataloader):
                self.get_activations(batch)
        self.store_cache()

    def eval(
        self,
        dataset: MixedData,
        batch_size: int = 64,
        **kwargs,
    ):
        # Check this explicitly because otherwise things can break in weird ways
        # when we assume that anomaly labels are included.
        assert isinstance(dataset, MixedData), type(dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        for batch in tqdm.tqdm(dataloader):
            # Remove anomaly labels
            batch = batch[0]
            self.get_activations(batch)

        self.store_cache()

        return {}

    def layerwise_scores(self, batch):
        raise NotImplementedError
