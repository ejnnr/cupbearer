from __future__ import annotations

import math
from abc import abstractmethod  # for postponed evaluation of annotations
from typing import overload

import torch
from torch import nn

from cupbearer import models


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
    # TODO: I think we should likely get rid of get_default and instead just have some
    # informal collection of helper functions for building reasonable abstractions.
    @classmethod
    @abstractmethod
    def get_default(cls, model: torch.nn.Module, size_reduction: int) -> Abstraction:
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

    @classmethod
    def get_default(
        cls,
        model: torch.nn.Module,
        size_reduction: int,
    ) -> LocallyConsistentAbstraction:
        def get_mlp_abstraction(
            model: models.MLP, size_reduction: int
        ) -> tuple[dict[str, nn.Module], dict[str, nn.Module]]:
            """Help method for MLP models"""
            tau_maps = {}
            steps = {}
            in_features = math.prod(model.input_shape)
            full_dims = model.hidden_dims + [model.output_dim]
            abstract_dims = [
                reduce_size(dim, size_reduction) for dim in model.hidden_dims
            ] + [model.output_dim]
            for i, (in_features, out_features) in enumerate(
                zip(abstract_dims[:-1], abstract_dims[1:])
            ):
                tau_maps[f"layers.linear_{i}.output"] = nn.Linear(
                    full_dims[i], in_features
                )
                # TODO: should potentially include ReLU here, but want to try
                # this version The i + 1 is needed because steps[name]
                # describes how to compute the *output* of the layer with name
                # `name`, which is the next one relative to the one we're
                # currently looking at.
                steps[f"layers.linear_{i + 1}.output"] = nn.Linear(
                    in_features, out_features
                )

            tau_maps[f"layers.linear_{len(abstract_dims) - 1}.output"] = nn.Identity()

            return tau_maps, steps

        if isinstance(model, models.MLP):
            tau_maps, steps = get_mlp_abstraction(model, size_reduction)
        elif isinstance(model, models.CNN):
            tau_maps = {}
            steps = {}
            abstract_dims = [reduce_size(dim, size_reduction) for dim in model.channels]
            for i, (in_features, out_features) in enumerate(
                zip(abstract_dims[:-1], abstract_dims[1:])
            ):
                tau_maps[f"conv_layers.conv_{i}.output"] = nn.Conv2d(
                    model.channels[i],
                    in_features,
                    model.kernel_sizes[i],
                    padding="same",
                )
                if i < len(abstract_dims) - 1:
                    # TODO: should potentially include ReLU here, but want to try this
                    steps[f"conv_layers.conv_{i + 1}.output"] = nn.Sequential(
                        nn.MaxPool2d(2),
                        nn.Conv2d(
                            in_features,
                            out_features,
                            model.kernel_sizes[i],
                            padding="same",
                        ),
                    )
            tau_maps[f"conv_layers.conv_{len(abstract_dims) - 1}.output"] = nn.Conv2d(
                model.channels[-1],
                abstract_dims[-1],
                model.kernel_sizes[-1],
                padding="same",
            )

            mlp_tau_maps, mlp_steps = get_mlp_abstraction(model.mlp, size_reduction)
            for k, v in mlp_tau_maps.items():
                tau_maps[f"mlp.{k}"] = v
                if k == "layers.linear_0.output":
                    next_mlp_dim = (
                        model.mlp.hidden_dims[0]
                        if model.mlp.hidden_dims
                        else model.mlp.output_dim
                    )
                    # Need to include a global pooling step here first
                    steps[f"mlp.{k}"] = nn.Sequential(
                        nn.AdaptiveMaxPool2d((1, 1)),
                        nn.Flatten(),
                        # Need to create a Linear layer here since from the perspective
                        # of the MLP abstraction, this is the step from input to first
                        # activation, which isn't represented.
                        nn.Linear(
                            abstract_dims[-1],
                            reduce_size(next_mlp_dim, size_reduction),
                        ),
                    )
                else:
                    steps[f"mlp.{k}"] = mlp_steps[k]
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        return cls(tau_maps, steps)


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

    @classmethod
    def get_default(
        cls,
        model: torch.nn.Module,
        size_reduction: int,
    ) -> AutoencoderAbstraction:
        def get_mlp_abstraction(
            model: models.MLP, size_reduction: int
        ) -> tuple[dict[str, nn.Module], dict[str, nn.Module]]:
            """Help method for MLP models"""
            tau_maps = {}
            decoders = {}
            abstract_dims = [
                reduce_size(dim, size_reduction) for dim in model.hidden_dims
            ] + [model.output_dim]
            i = -1
            for i, (activation_dim, abstract_dim) in enumerate(
                zip(model.hidden_dims, abstract_dims)
            ):
                tau_maps[f"layers.linear_{i}.output"] = nn.Linear(
                    activation_dim, abstract_dim
                )
                decoders[f"layers.linear_{i}.output"] = nn.Linear(
                    abstract_dim, activation_dim
                )
                # TODO: this is a bit too basic probably

            # Let autoencoder be trivial for output layer
            tau_maps[f"layers.linear_{i + 1}.output"] = nn.Identity()
            decoders[f"layers.linear_{i + 1}.output"] = nn.Identity()

            return tau_maps, decoders

        if isinstance(model, models.MLP):
            tau_maps, decoders = get_mlp_abstraction(model, size_reduction)
        elif isinstance(model, models.CNN):
            tau_maps = {}
            decoders = {}
            abstract_dims = [reduce_size(dim, size_reduction) for dim in model.channels]
            for i, (activation_dim, abstract_dim) in enumerate(
                zip(model.channels, abstract_dims)
            ):
                tau_maps[f"conv_layers.conv_{i}.output"] = nn.Conv2d(
                    activation_dim,
                    abstract_dim,
                    model.kernel_sizes[i],
                    padding="same",
                )
                decoders[f"conv_layers.conv_{i}.output"] = nn.Conv2d(
                    abstract_dim,
                    activation_dim,
                    model.kernel_sizes[i],
                    padding="same",
                )
                # TODO: this is a bit too basic probably

            mlp_tau_maps, mlp_decoders = get_mlp_abstraction(model.mlp, size_reduction)
            for k, v in mlp_tau_maps.items():
                tau_maps[f"mlp.{k}"] = v
                decoders[f"mlp.{k}"] = mlp_decoders[k]
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        return cls(tau_maps, decoders)
