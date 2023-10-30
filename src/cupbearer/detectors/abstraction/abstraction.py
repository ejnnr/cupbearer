import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from cupbearer.models.hooked_model import HookedModel


def assert_acyclic(graph: dict[str, list[str]]):
    visited = set()
    current_path = set()

    def visit(node):
        if node in visited:
            return True, None
        visited.add(node)
        current_path.add(node)
        for parent in graph[node]:
            if parent in current_path or visit(parent):
                return False, current_path
        current_path.remove(node)
        return True, None

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
        # Check that the parents graph is acyclic:
        assert_acyclic(parents)

        super().__init__()
        self.parents = parents
        self.tau_maps = nn.ModuleDict(tau_maps)
        self.steps = nn.ModuleDict(steps)

    def forward(
        self, activations: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        abstractions = {
            name: tau_map(activations[name]) for name, tau_map in self.tau_maps.items()
        }

        predicted_abstractions: dict[str, torch.Tensor | None] = {}

        def compute(node: str):
            try:
                return predicted_abstractions[node]
            except KeyError:
                if self.parents[node] == []:
                    # This is a root node, so we can't compute a prediction for it.
                    return None
                return self.steps[node](
                    *[compute(parent) for parent in self.parents[node]]
                )

        for node in self.parents:
            predicted_abstractions[node] = compute(node)

        return abstractions, predicted_abstractions


def abstraction_collate(model: HookedModel, names=None, return_original_batch=False):
    """Create a collate function that turns a dataloader into one for activations.

    You can use the output of this function as the collate_fn argument to any
    dataloader. Then the output of that dataloader won't be the data itself,
    but instead the outputs and activations of `model` on that data.

    Effictively, this isn't really a collate_fn, think of it more as a (pretty extreme)
    transform. The reason to implement this as a collate_fn is that we want to transform
    entire batches at once instead of running a forward pass of `model` on every
    single example.

    See https://github.com/pytorch/vision/issues/157#issuecomment-431289943 for the
    inspiration for this trick.

    WARNING: This function assumes that the dataset it is used for returns
    tuples/lists where the first element is the input to the model. (The other elements
    are ignored.)

    Args:
        model: The model to run on the data.
        names: names of activations to return. If None, all activations are returned.

    Returns:
        A function that takes a batch of data (which hasn't been collated yet)
        and returns a list of [logits, activations, original_batch], where logits is an
        array of shape (batch_size, *output_shape), activations is a list of arrays
        of shape (batch_size, *activation_shape), and original_batch is the collated
        version of the data itself (only if `return_original_batch` is True).
    """

    def mycollate(batch):
        collated = default_collate(batch)
        if isinstance(collated, (tuple, list)):
            inputs = collated[0]
        else:
            inputs = collated
        logits, activations = model.get_activations(inputs, names)
        if return_original_batch:
            return [logits, activations, collated]
        return [logits, activations]

    return mycollate
