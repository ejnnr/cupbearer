from typing import Any, Callable

import torch

from cupbearer import utils
from cupbearer.detectors.extractors.core import DictionaryExtractor


class ActivationExtractor(DictionaryExtractor):
    def __init__(
        self,
        names: list[str],
        return_inputs: bool = False,
        individual_processing_fn: Callable[[torch.Tensor, Any, str], torch.Tensor]
        | None = None,
        global_processing_fn: Callable[
            [dict[str, torch.Tensor]], dict[str, torch.Tensor]
        ]
        | None = None,
    ):
        super().__init__(
            feature_names=names,
            individual_processing_fn=individual_processing_fn,
            global_processing_fn=global_processing_fn,
        )
        self.names = names
        self.return_inputs = return_inputs

    def __call__(self, batch: Any) -> dict[str, torch.Tensor]:
        inputs = utils.inputs_from_batch(batch)
        acts = utils.get_activations(inputs, model=self.model, names=self.names)
        if self.return_inputs:
            return {**acts, "inputs": inputs}
        return acts
