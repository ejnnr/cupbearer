# An abstraction of a fixed computational graph (like a neural network) will
# take in a sequence of activations (with a fixed length). It learns one linear
# map for each activation, mapping it to an abstract representation. Additionally,
# it learns a non-linear map from each abstract representation to the next one.
# This non-linear map should output a *distribution* over the next abstract
# representation. Finally, it learns a linear map from the last abstract
# representation to the output of the computation.
# All of this is encapsulated in a flax module.

import functools
from typing import Callable, List, Sequence
import flax.linen as nn
import jax
from iceberg import Drawable, Bounds, Colors, Color, Corner, PathStyle
from iceberg.primitives import Arrange, Ellipse, Padding, Line, Compose
from abstractions.computations import Computation, Orientation, Step
from abstractions import data


def draw_computation(
    computation: Computation, return_nodes=False, layer_scores=None
) -> Drawable:
    steps = [step.get_drawable() for step in computation]
    nodes = [
        Ellipse(Bounds(size=(50, 50)), border_color=Colors.BLACK) for _ in computation
    ]
    if layer_scores is not None:
        # The first node doesn't incur any loss in the current implementation,
        # so we don't color it.
        nodes[0].fill_color = Color(0.3, 0.3, 0.3)

        # TODO: might not make sense for all loss functions
        min_score = 0
        max_score = max(layer_scores)
        normalized_scores = [
            (score - min_score) / (max_score - min_score) for score in layer_scores
        ]

        assert len(nodes) - 1 == len(normalized_scores), (
            f"len(nodes) - 1 = {len(nodes) - 1}"
            f"!= len(normalized_scores) = {len(normalized_scores)}"
        )
        for node, score in zip(nodes[1:], normalized_scores):
            node.fill_color = Color(1, 0, 0, score)

    # interleave the two lists:
    drawables = [x for pair in zip(steps, nodes) for x in pair]
    arranged = Arrange(drawables, gap=100)

    lines = []
    linestyle = PathStyle(color=Colors.BLACK)

    for a, b in zip(drawables[:-1], drawables[1:]):
        start = arranged.child_bounds(a).corners[Corner.MIDDLE_RIGHT]
        end = arranged.child_bounds(b).corners[Corner.MIDDLE_LEFT]
        line = Line(start, end, linestyle)
        lines.append(line)

    res = Compose((*lines, arranged))

    if return_nodes:
        return res, nodes
    return res


class Model(nn.Module):
    computation: Computation

    @nn.compact
    def __call__(self, x, return_activations=False, train=True):
        activations = []
        *main_maps, output_map = self.computation
        for step in main_maps:
            x = step(x)
            activations.append(x)

        x = output_map(x)

        if return_activations:
            return x, activations
        return x

    def get_drawable(self, return_nodes=False, layer_scores=None) -> Drawable:
        return draw_computation(self.computation, return_nodes, layer_scores)


class Abstraction(nn.Module):
    computation: Computation
    abstraction_maps: List[Step]

    @nn.compact
    def __call__(self, activations: List[jax.Array], train: bool = True):
        abstractions = [
            abstraction_map(x)
            for abstraction_map, x in zip(self.abstraction_maps, activations)
        ]

        input_map, *main_maps, output_map = self.computation
        *main_abstractions, output = abstractions
        predicted_abstractions = [
            step(x) for step, x in zip(main_maps, main_abstractions)
        ]

        predicted_output = output_map(output)

        return abstractions, predicted_abstractions, predicted_output

    def get_drawable(self, full_model: Model, layer_scores=None) -> Drawable:
        model_drawable, model_nodes = full_model.get_drawable(return_nodes=True)
        abstraction_drawable, abstraction_nodes = draw_computation(
            self.computation, return_nodes=True, layer_scores=layer_scores
        )

        VERTICAL_DISTANCE = 250

        both_computations = Arrange(
            [model_drawable, abstraction_drawable],
            Arrange.Direction.VERTICAL,
            gap=VERTICAL_DISTANCE,
        )

        tau_maps = [
            map.get_drawable(orientation=Orientation.VERTICAL)
            for map in self.abstraction_maps
        ]

        lines = []
        linestyle = PathStyle(color=Colors.BLACK)
        positioned_tau_maps = []

        for model_node, tau_map, abstract_node in zip(
            model_nodes, tau_maps, abstraction_nodes
        ):
            model_bounds = both_computations.child_bounds(model_node)
            abstract_bounds = both_computations.child_bounds(abstract_node)
            # Want to align centers:
            x = model_bounds.center[0] - tau_map.bounds.width / 2
            y = (
                model_bounds.bottom + abstract_bounds.top
            ) / 2 - tau_map.bounds.height / 2
            tau_map = tau_map.move(x, y)
            positioned_tau_maps.append(tau_map)

            # Line from model node to tau map:
            start = model_bounds.corners[Corner.BOTTOM_MIDDLE]
            end = tau_map.bounds.corners[Corner.TOP_MIDDLE]
            lines.append(Line(start, end, linestyle))

            # Line from tau map to abstraction node:
            start = tau_map.bounds.corners[Corner.BOTTOM_MIDDLE]
            end = abstract_bounds.corners[Corner.TOP_MIDDLE]
            lines.append(Line(start, end, linestyle))

        return Compose((*lines, *positioned_tau_maps, both_computations))


def abstraction_collate(model: nn.Module, params, return_original_batch=False):
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
        model: The model to run on the data. It's call method must have a
            `return_activations` argument and return a tuple of (logits, activations)
            if `return_activations` is True.
        params: The parameters of the model to use.

    Returns:
        A function that takes a batch of data (which hasn't been collated yet)
        and returns a list of [logits, activations, original_batch], where logits is an
        array of shape (batch_size, *output_shape), activations is a list of arrays
        of shape (batch_size, *activation_shape), and original_batch is the collated
        version of the data itself (only if `return_original_batch` is True).
    """
    forward_fn = jax.jit(
        lambda x: model.apply({"params": params}, x, return_activations=True)
    )

    def mycollate(batch):
        collated = data.numpy_collate(batch)
        inputs = collated[0]
        logits, activations = forward_fn(inputs)
        if return_original_batch:
            return [logits, activations, collated]
        return [logits, activations]

    return mycollate
