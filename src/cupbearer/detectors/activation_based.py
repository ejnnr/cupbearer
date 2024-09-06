from typing import Any, Callable

import torch

from .anomaly_detector import AnomalyDetector
from .extractors import ActivationExtractor, FeatureCache, FeatureExtractor


class ActivationBasedDetector(AnomalyDetector):
    """Base class for detectors that defaults to an activation feature extractor.

    The AnomalyDetector base class uses the identity feature extractor by default,
    this one just changes that and exposes arguments for customizing an activation
    feature extractor. The feature extractor can still be overriden.
    """

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
        processed_names: list[str] | None = None,
        layer_aggregation: str = "mean",
        cache: FeatureCache | None = None,
    ):
        if feature_extractor is None:
            if activation_names is None:
                raise ValueError(
                    "Either a feature extractor or a list of activation names "
                    "must be provided."
                )
            feature_extractor = ActivationExtractor(
                names=activation_names,
                individual_processing_fn=individual_processing_fn,
                global_processing_fn=global_processing_fn,
                processed_names=processed_names,
                cache=cache,
            )
        super().__init__(
            feature_extractor=feature_extractor, layer_aggregation=layer_aggregation
        )
