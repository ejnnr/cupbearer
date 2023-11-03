import math
from typing import overload

import torch
from torch import nn

from cupbearer import models
from cupbearer.models.hooked_model import HookedModel


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


def mlp_abstraction(
    model: models.MLP, size_reduction: int
) -> tuple[dict[str, nn.Module], dict[str, nn.Module]]:
    tau_maps = {}
    steps = {}
    in_features = math.prod(model.input_shape)
    full_dims = model.hidden_dims + [model.output_dim]
    abstract_dims = [reduce_size(dim, size_reduction) for dim in model.hidden_dims] + [
        model.output_dim
    ]
    for i, (in_features, out_features) in enumerate(
        zip(abstract_dims[:-1], abstract_dims[1:])
    ):
        if i < len(abstract_dims):
            tau_maps[f"post_linear_{i}"] = nn.Linear(full_dims[i], in_features)
        else:
            tau_maps[f"post_linear_{i}"] = nn.Identity()
        if i > 0:
            # TODO: should potentially include ReLU here, but want to try this version
            steps[f"post_linear_{i}"] = nn.Linear(in_features, out_features)

    return tau_maps, steps


def get_default_abstraction(model: HookedModel, size_reduction: int) -> Abstraction:
    if isinstance(model, models.MLP):
        tau_maps, steps = mlp_abstraction(model, size_reduction)
    elif isinstance(model, models.CNN):
        tau_maps = {}
        steps = {}
        abstract_dims = [reduce_size(dim, size_reduction) for dim in model.channels]
        for i, (in_features, out_features) in enumerate(
            zip(abstract_dims[:-1], abstract_dims[1:])
        ):
            tau_maps[f"post_conv_{i}"] = nn.Conv2d(
                model.channels[i],
                in_features,
                model.kernel_sizes[i],
                padding="same",
            )
            if i > 0:
                # TODO: should potentially include ReLU here, but want to try this
                steps[f"post_conv_{i}"] = nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(
                        in_features, out_features, model.kernel_sizes[i], padding="same"
                    ),
                )

        mlp_tau_maps, mlp_steps = mlp_abstraction(model.mlp, size_reduction)
        for k, v in mlp_tau_maps.items():
            tau_maps[f"mlp_{k}"] = v
            if k == "post_linear_0":
                # Need to include a global pooling step here first
                steps[f"mlp_{k}"] = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten(), mlp_steps[k]
                )
            else:
                steps[f"mlp_{k}"] = mlp_steps[k]
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    return Abstraction(tau_maps, steps)
