# An abstraction of a fixed computational graph (like a neural network) will
# take in a sequence of activations (with a fixed length). It learns one linear
# map for each activation, mapping it to an abstract representation. Additionally,
# it learns a non-linear map from each abstract representation to the next one.
# This non-linear map should output a *distribution* over the next abstract
# representation. Finally, it learns a linear map from the last abstract
# representation to the output of the computation.
# All of this is encapsulated in a flax module.

import functools
from typing import Callable, List, Optional

import flax.linen as nn
import jax

from cupbearer.data import _shared
from cupbearer.models.computations import (
    Computation,
    Conv,
    ConvBlock,
    DenseBlock,
    Linear,
    Model,
    Orientation,
    ReluLinear,
    SoftmaxDrop,
    Step,
    draw_computation,
    make_image_grid,
    reduce_size,
)


class Wrapper(nn.Module):
    """A Module that wraps a function.

    The intended use is grouping multiple submodules under a single name
    in Modules that use nn.compact. See the Abstraction class for an example.
    """

    func: Callable

    @nn.compact
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Abstraction(nn.Module):
    computation: Computation
    tau_maps: List[Step]

    @nn.compact
    def __call__(self, activations: List[jax.Array], train: bool = True):
        assert len(activations) == len(
            self.tau_maps
        ), f"Got {len(activations)} activations but {len(self.tau_maps)} tau maps"

        if not isinstance(self.tau_maps[0], nn.Module):
            # This is just a hack to put the parameters of all the tau maps
            # under a single "tau_maps" key in the params dict.
            abstractions = Wrapper(
                name="tau_maps",  # type: ignore
                func=lambda activations: [
                    tau_map(x) for tau_map, x in zip(self.tau_maps, activations)
                ],
            )(activations)
        else:
            # If the tau maps are flax modules, flax will already group them.
            abstractions = [
                tau_map(x) for tau_map, x in zip(self.tau_maps, activations)
            ]

        input_map, *maps = self.computation
        assert len(maps) == len(abstractions)
        if not isinstance(maps[0], nn.Module):
            predicted_abstractions = Wrapper(
                name="computational_steps",  # type: ignore
                func=lambda abstractions: [
                    step(x) for step, x in zip(maps, abstractions)
                ],
            )(abstractions)
        else:
            predicted_abstractions = [step(x) for step, x in zip(maps, abstractions)]

        predicted_output = predicted_abstractions[-1]

        return abstractions, predicted_abstractions[:-1], predicted_output

    def get_drawable(self, full_model: Model, layer_scores=None, inputs=None):
        try:
            from iceberg import Colors, Corner, PathStyle
            from iceberg.arrows import Arrow
            from iceberg.primitives import (
                Arrange,
                Compose,
                Directions,
                Line,
            )
        except ImportError:
            raise ImportError(
                "You need to install iceberg-dsl to use the abstraction visualization"
            )

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
            map.get_drawable(orientation=Orientation.VERTICAL) for map in self.tau_maps
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
            lines.append(Arrow(start, end, linestyle))

        res = Compose((*lines, *positioned_tau_maps, both_computations))

        if inputs is not None:
            grid = make_image_grid(inputs)
            res = res.next_to(grid, Directions.LEFT * 20)

        return res


def AxisAlignedAbstraction(computation: Computation, tau_maps: List[Step]):
    output_dims = [tau.output_dim for tau in tau_maps]
    print(output_dims)
    print("Computation:", [step.output_dim for step in computation])
    tau_maps = [SoftmaxDrop(output_dim=output_dim) for output_dim in output_dims]
    return Abstraction(computation=computation, tau_maps=tau_maps)


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
        collated = _shared.numpy_collate(batch)
        if isinstance(collated, (tuple, list)):
            inputs = collated[0]
        else:
            inputs = collated
        logits, activations = forward_fn(inputs)
        if return_original_batch:
            return [logits, activations, collated]
        return [logits, activations]

    return mycollate


def get_tau_map(step: Step, kernel_init=nn.linear.default_kernel_init) -> Step:
    match step:
        case Linear(output_dim=dim, kernel_init=_):
            return Linear(output_dim=dim, kernel_init=kernel_init)
        case ReluLinear(output_dim=dim, flatten_input=_):
            return Linear(output_dim=dim)
        case Conv(output_dim=dim, kernel_init=_):
            return Conv(output_dim=dim, kernel_init=kernel_init)
        case ConvBlock(output_dim=dim):
            return Conv(output_dim=dim, kernel_init=kernel_init)
        case DenseBlock(output_dim=dim, is_first=_):
            return Linear(output_dim=dim, kernel_init=kernel_init)
        case _:
            raise ValueError(f"Unknown step type {step}")


def get_tau_maps(
    computation: Computation, kernel_init=nn.linear.default_kernel_init
) -> list[Step]:
    """Get a list of abstraction maps from a computation."""
    # We don't want a tau map for the final step, that just produces the output,
    # which we compare directly.
    return list(
        map(functools.partial(get_tau_map, kernel_init=kernel_init), computation[:-1])
    )


def get_default_abstraction(
    model: Model,
    size_reduction: int,
    output_dim: Optional[int] = None,
    abstraction_cls=Abstraction,
) -> Abstraction:
    """Get a sensible default abstraction for a model.

    `size_reduction` is the factor by which hidden dimensions in the abstraction should
    be smaller. `output_dim` is the output dimension of the abstraction. If None, the
    model's output dimension is used.
    """
    abstract_computation = reduce_size(model.computation, size_reduction, output_dim)
    tau_maps = get_tau_maps(abstract_computation)
    return abstraction_cls(computation=abstract_computation, tau_maps=tau_maps)
