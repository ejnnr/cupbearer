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


def assert_acyclic(graph: dict[str, list[str]]):
    visited = set()
    current_path = set()

    def visit(node):
        if node in visited:
            return False, None
        visited.add(node)
        current_path.add(node)
        for parent in graph[node]:
            if parent in current_path:
                return True, current_path
            found_cycle, cycle = visit(parent)
            if found_cycle:
                return True, cycle
        current_path.remove(node)
        return False, None

    for vertex in graph:
        found_cycle, cycle = visit(vertex)
        if found_cycle:
            raise ValueError(f"Cycle found in activation graph: {cycle}")


class Abstraction(nn.Module):
    pass


class LocallyConsistentAbstraction(Abstraction):
    def __init__(
        self,
        tau_maps: dict[str, nn.Module],
        steps: dict[str, nn.Module],
        parents: dict[str, list[str]] | None = None,
    ):
        if parents is None:
            # default is a linear graph:
            names = list(tau_maps.keys())
            parents = {name: [prev] for prev, name in zip(names[:-1], names[1:])}
            parents[names[0]] = []
        # Check that the parents graph is acyclic:
        assert_acyclic(parents)

        super().__init__()
        self.parents = parents
        self.tau_maps = nn.ModuleDict(tau_maps)
        self.steps = nn.ModuleDict(steps)

    def forward(
        self, activations: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        # TODO: given that this is called in each forward pass, it might be worth
        # switching to https://github.com/metaopt/optree for faster tree maps
        abstractions = {
            name: tau_map(activations[name]) for name, tau_map in self.tau_maps.items()
        }

        predicted_abstractions: dict[str, torch.Tensor | None] = {
            name: self.steps[name](*(abstractions[parent] for parent in parents))
            if parents
            else None
            for name, parents in self.parents.items()
        }

        return abstractions, predicted_abstractions


class AutoencoderAbstraction(Abstraction):
    def __init__(
        self,
        tau_maps: dict[str, nn.Module],  # encoders
        decoders: dict[str, nn.Module],  # decoders
    ):
        super().__init__()
        assert tau_maps.keys() == decoders.keys()
        for name in tau_maps:
            if "/" in name:
                raise ValueError(
                    f"For technical reasons, names cant't contain '/', got {name}"
                )
        # Pytorch's ModuleDict doesn't allow '.' in keys, so we replace them with '/'
        tau_maps = {
            name.replace(".", "/"): tau_map for name, tau_map in tau_maps.items()
        }
        decoders = {
            name.replace(".", "/"): decoder for name, decoder in decoders.items()
        }
        self.tau_maps = nn.ModuleDict(tau_maps)
        self.decoders = nn.ModuleDict(decoders)

    def forward(
        self, activations: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        # Same name replacement as in __init__, to make user-facing names consistent.
        activations = {
            name.replace(".", "/"): activation
            for name, activation in activations.items()
        }
        abstractions = {
            name: tau_map(activations[name]) for name, tau_map in self.tau_maps.items()
        }

        reconstructed_activations: dict[str, torch.Tensor | None] = {
            name.replace("/", "."): self.decoders[name](abstraction)
            for name, abstraction in abstractions.items()
        }

        abstractions = {
            name.replace("/", "."): abstraction
            for name, abstraction in abstractions.items()
        }

        return abstractions, reconstructed_activations
