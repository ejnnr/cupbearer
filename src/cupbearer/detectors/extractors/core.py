from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from cupbearer import utils
from cupbearer.data import MixedData


class FeatureExtractor(ABC):
    def set_model(self, model: torch.nn.Module):
        self.model = model

    @abstractmethod
    def __call__(self, batch: Any) -> Any:
        pass


class FeatureCache:
    """Cache for features to speed up using multiple anomaly detectors.

    The main use case for this is if the model is expensive to run and we want to try
    many different detectors that require similar features.

    The cache stores a dict from (input, feature_name) to the features.

    WARNING: This cache is not safe to use across different models or
    preprocessing functions! It does not attempt to track those, and using the same
    cache with different model/preprocessors will likely result in incorrect results.

    Args:
        device: The device to load the features to (same as model and data)
        storage_device: The device to store the features on. This should be a
            device with more memory than `device`, since the features are stored
            in the cache. Default is "cpu".
    """

    def __init__(self, device: str, storage_device: str = "cpu"):
        """Create an empty cache."""
        self.cache: dict[tuple[Any, str], torch.Tensor] = {}
        self.device = device
        self.storage_device = storage_device
        # Just for debugging purposes:
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self.cache)

    def __contains__(self, key):
        return key in self.cache

    def count_missing(self, dataset: Dataset, feature_names: list[str]):
        """Count how many inputs from `dataset` are missing from the cache.

        `feature_names` is a list of the names of the features we need to be
        in the cache. An input counts as missing if *some* of the features are
        missing for that input.
        """
        count = 0
        for sample in dataset:
            if isinstance(dataset, MixedData) and dataset.return_anomaly_labels:
                sample = sample[0]
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                sample = sample[0]
            if not all((sample, name) in self.cache for name in feature_names):
                count += 1
        return count

    def store(self, path: str | Path):
        utils.save(self.cache, path)

    @classmethod
    def load(cls, path: str | Path, device: str, storage_device: str = "cpu"):
        cache = cls(device=device, storage_device=storage_device)
        cache.cache = utils.load(path)
        return cache

    def get_features(
        self,
        inputs,
        feature_names: list[str],
        feature_fn: Callable[[Any], dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Get features for a batch of inputs, using the cache if possible.

        If any features are missing from the cache, they are computed and added
        to the cache.

        Args:
            inputs: The inputs to get features for.
            feature_names: The names of the features to get. This is used to check
                whether all features are already in the cache.
            feature_fn: Takes in `inputs` and returns a dictionary of features.

        Returns:
            A dict from feature name to the features.
        """
        # We want to handle cases where some but not all elements are in the cache.
        missing_indices = []
        results: dict[str, list[torch.Tensor | None]] = defaultdict(
            lambda: [None] * len(inputs)
        )

        for i, input in enumerate(inputs):
            # convert input to tuple for hashing if tensor
            if isinstance(input, torch.Tensor):
                input = utils.tensor_to_tuple(input)
            # The keys into the cache contain the input and the name of the feature.
            keys = [(input, name) for name in feature_names]
            # In principle we could support the case where some but not all features
            # for a given input are already in the cache. If the missing features
            # are early in the model, this might save some time since we wouldn't
            # have to do the full forward pass. But that seems like a niche use case
            # and not worth the added complexity. So for now, we recompute all
            # features on inputs where some features are missing.
            if all(key in self.cache for key in keys):
                self.hits += 1
                for name in feature_names:
                    results[name][i] = self.cache[(input, name)].to(self.device)
            else:
                missing_indices.append(i)

        if not missing_indices:
            return {name: torch.stack(results[name]) for name in feature_names}

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

        new_features = feature_fn(inputs)
        self.misses += len(inputs)

        # Fill in the missing features
        for name, feature in new_features.items():
            for i, idx in enumerate(missing_indices):
                results[name][idx] = feature[i]
                input = inputs[i]
                if isinstance(input, torch.Tensor):
                    input = utils.tensor_to_tuple(input)
                self.cache[(input, name)] = feature[i].to(self.storage_device)

        del new_features  # free up device memory

        assert all(
            all(result is not None for result in results[name])
            for name in feature_names
        )

        return {name: torch.stack(results[name]) for name in feature_names}


class DictionaryExtractor(FeatureExtractor):
    @abstractmethod
    def __call__(self, batch: Any) -> dict[str, torch.Tensor]:
        pass

    def __init__(
        self,
        feature_names: list[str],
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        global_processing_fn: Callable[
            [dict[str, torch.Tensor]], dict[str, torch.Tensor]
        ]
        | None = None,
        cache: FeatureCache | None = None,
    ):
        self.feature_names = feature_names
        self.individual_processing_fn = individual_processing_fn
        self.global_processing_fn = global_processing_fn
        self.cache = cache

    def _get_features_no_cache(self, inputs) -> dict[str, torch.Tensor]:
        device = next(self.model.parameters()).device
        inputs = utils.inputs_to_device(inputs, device)
        features = self(inputs)

        # Can be used to for example select activations at specific token positions
        if self.individual_processing_fn is not None:
            features = {
                k: self.individual_processing_fn(v, inputs, k)
                for k, v in features.items()
            }

        if self.global_processing_fn is not None:
            features = self.global_processing_fn(features)

        return features

    def get_features(self, batch) -> dict[str, torch.Tensor]:
        inputs = utils.inputs_from_batch(batch)

        if self.cache is None:
            return self._get_features_no_cache(inputs)

        return self.cache.get_features(
            inputs, self.feature_names, self._get_features_no_cache
        )


class IdentityExtractor(FeatureExtractor):
    """Extractor that returns the input as is."""

    def __call__(self, batch: Any) -> Any:
        return batch
