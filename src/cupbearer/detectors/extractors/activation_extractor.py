from typing import Any, Callable

import torch

from cupbearer import utils

from .core import FeatureCache, FeatureExtractor


class ActivationExtractor(FeatureExtractor):
    def __init__(
        self,
        names: list[str],
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        global_processing_fn: Callable[
            [dict[str, torch.Tensor]], dict[str, torch.Tensor]
        ]
        | None = None,
        cache: FeatureCache | None = None,
    ):
        super().__init__(
            feature_names=names,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
            cache=cache,
        )
        self.names = names

    def compute_features(self, inputs: Any) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        return utils.get_activations(inputs, model=self.model, names=self.names)
