from __future__ import annotations

import math
from typing import overload

import torch
from torch import nn


@overload
def reduce_size(shape: int, size_reduction: int) -> int:
    ...


@overload
def reduce_size(shape: tuple[int, ...], size_reduction: int) -> tuple[int, ...]:
    ...


def reduce_size(
    shape: int | tuple[int, ...], size_reduction: int
) -> int | tuple[int, ...]:
    if isinstance(shape, int):
        return math.ceil(shape / size_reduction)
    return tuple(math.ceil(x / size_reduction) for x in shape)


class Abstraction(nn.Module):
    tau_maps: _ModuleDict

    def forward(self, inputs, activations: dict[str, torch.Tensor]):
        """Get the abstractions for the given inputs and activations.

        Args:
            inputs: The inputs to the model.
            activations: A dictionary mapping activation names to the activations of the
                full model at that name.
        """
        raise NotImplementedError


class _ModuleDict(nn.Module):
    """A ModuleDict that allows '.' in keys (but not '/').

    Pytorch's ModuleDict doesn't allow '.' in keys, but our activation names contain
    a lot of periods.
    """

    def __init__(self, modules: dict[str, nn.Module]):
        super().__init__()
        for name in modules:
            if "/" in name:
                raise ValueError(
                    f"For technical reasons, names cant't contain '/', got {name}"
                )
        # Pytorch's ModuleDict doesn't allow '.' in keys, so we replace them with '/'
        modules = {name.replace(".", "/"): module for name, module in modules.items()}
        self.dict = nn.ModuleDict(modules)

    def __getitem__(self, key: str) -> nn.Module:
        return self.dict[key.replace(".", "/")]

    def __len__(self) -> int:
        return len(self.dict)

    def __iter__(self):
        for key in self.dict:
            yield key.replace("/", ".")

    def __contains__(self, key: str) -> bool:
        return key.replace(".", "/") in self.dict

    def items(self):
        for key, value in self.dict.items():
            yield key.replace("/", "."), value

    def values(self):
        return self.dict.values()

    def keys(self):
        # HACK: we want to return an actual dict_keys object to make sure that
        # e.g. equality with other dict_keys objects works as expected. So we create
        # a dummy dictionary and return its keys.
        # BUG: This probably still doesn't behave like a real dict_keys object.
        # For example, if the underlying dictionary changes, this won't reflect that.
        return {key.replace("/", "."): None for key in self.dict}.keys()


class LocallyConsistentAbstraction(Abstraction):
    def __init__(
        self,
        tau_maps: dict[str, nn.Module],
        abstract_model: nn.Module,
    ):
        super().__init__()
        self.tau_maps = _ModuleDict(tau_maps)
        self.abstract_model = abstract_model

    def forward(
        self, inputs, activations: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        # TODO: given that this is called in each forward pass, it might be worth
        # switching to https://github.com/metaopt/optree for faster tree maps
        assert activations.keys() == self.tau_maps.keys()
        abstractions = {
            name: tau_map(activations[name]) for name, tau_map in self.tau_maps.items()
        }

        predicted_abstractions = {}

        # This is similar to `get_activations`, except we also intervene in the forward
        # pass. Maybe we could unify them?
        def make_hook(name, is_input):
            def hook(module, input, output):
                # First, we cache the value that the abstract model predicted here:
                if is_input:
                    predicted_abstractions[name] = (
                        input if isinstance(input, torch.Tensor) else input[0]
                    )
                else:
                    predicted_abstractions[name] = output

                # Then, we replace it with the "true" value (i.e. the one from the
                # tau map) for the rest of the forward pass. (This is what makes this
                # a *local* abstraction.)
                return abstractions[name]

            return hook

        hooks = []

        try:
            for name, module in self.abstract_model.named_modules():
                if name + ".input" in self.tau_maps:
                    hooks.append(
                        module.register_forward_hook(make_hook(name + ".input", True))
                    )
                if name + ".output" in self.tau_maps:
                    hooks.append(
                        module.register_forward_hook(make_hook(name + ".output", False))
                    )

            self.abstract_model(inputs)

        finally:
            for hook in hooks:
                hook.remove()

        assert abstractions.keys() == predicted_abstractions.keys()

        return abstractions, predicted_abstractions


class AutoencoderAbstraction(Abstraction):
    def __init__(
        self,
        tau_maps: dict[str, nn.Module],  # encoders
        decoders: dict[str, nn.Module],  # decoders
    ):
        super().__init__()
        assert tau_maps.keys() == decoders.keys()
        self.tau_maps = _ModuleDict(tau_maps)
        self.decoders = _ModuleDict(decoders)

    def forward(
        self, inputs, activations: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        abstractions = {
            name: tau_map(activations[name]) for name, tau_map in self.tau_maps.items()
        }

        reconstructed_activations: dict[str, torch.Tensor | None] = {
            name: self.decoders[name](abstraction)
            for name, abstraction in abstractions.items()
        }

        return abstractions, reconstructed_activations
