from typing import Callable

import torch


class _Finished(Exception):
    pass


def get_activations(
    model: torch.nn.Module, names: list[str], *args, **kwargs
) -> dict[str, torch.Tensor]:
    """Get the activations of the model for the given inputs.

    Args:
        model: The model to get the activations from.
        names: The names of the modules to get the activations from. Should be a list
            of strings corresponding to pytorch module names, with ".input" or ".output"
            appended to the end of the name to specify whether to get the input
            or output of the module.
        *args: Arguments to pass to the model.
        **kwargs: Keyword arguments to pass to the model.

    Returns:
        A dictionary mapping the names of the modules to the activations of the model
        at that module. Keys contain ".input" or ".output" just like `names`.
    """
    activations = {}
    hooks = []

    try:
        all_module_names = [name for name, _ in model.named_modules()]

        for name in names:
            assert name.endswith(".input") or name.endswith(
                ".output"
            ), f"Invalid name {name}, names should end with '.input' or '.output'"
            base_name = ".".join(name.split(".")[:-1])
            assert (
                base_name in all_module_names
            ), f"{base_name} is not a submodule of the model"

        def make_hook(name, is_input):
            def hook(module, input, output):
                if is_input:
                    if isinstance(input, torch.Tensor):
                        activations[name] = input
                    elif isinstance(input[0], torch.Tensor):
                        activations[name] = input[0]
                    else:
                        raise ValueError(
                            "Expected input to be a tensor or tuple with tensor as "
                            f"first element, got {type(input)}"
                        )
                else:
                    activations[name] = output

                if set(names).issubset(activations.keys()):
                    # HACK: stop the forward pass to save time
                    raise _Finished()

            return hook

        for name, module in model.named_modules():
            if name + ".input" in names:
                hooks.append(
                    module.register_forward_hook(make_hook(name + ".input", True))
                )
            if name + ".output" in names:
                hooks.append(
                    module.register_forward_hook(make_hook(name + ".output", False))
                )
        with torch.no_grad():
            try:
                model(*args, **kwargs)
            except _Finished:
                pass
    finally:
        # Make sure we always remove hooks even if an exception is raised
        for hook in hooks:
            hook.remove()

    return activations


def get_activations_and_grads(
    model: torch.nn.Module,
    names: list[str],
    output_func: Callable[[torch.Tensor], torch.Tensor],
    *args,
    **kwargs,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Get the activations and gradients of the model for the given inputs.

    Args:
        See `get_activations`.
        output_func: A function that takes the output of the model and reduces
            to a (batch_size, ) shaped tensor.

    Returns:
        `(activations, gradients)` where both are dictionaries as in `get_activations`.
    """
    activations = {}
    gradients = {}
    hooks = []

    try:
        all_module_names = [name for name, _ in model.named_modules()]

        for name in names:
            assert name.endswith(".input") or name.endswith(
                ".output"
            ), f"Invalid name {name}, names should end with '.input' or '.output'"
            base_name = ".".join(name.split(".")[:-1])
            assert (
                base_name in all_module_names
            ), f"{base_name} is not a submodule of the model"

        def make_hooks(name, is_input):
            def forward_hook(module, input, output):
                if is_input:
                    if isinstance(input, torch.Tensor):
                        activations[name] = input
                    elif isinstance(input[0], torch.Tensor):
                        activations[name] = input[0]
                    else:
                        raise ValueError(
                            "Expected input to be a tensor or tuple with tensor as "
                            f"first element, got {type(input)}"
                        )
                else:
                    activations[name] = output

            def backward_hook(module, grad_input, grad_output):
                if isinstance(grad_input, tuple):
                    grad_input, *_ = grad_input
                if isinstance(grad_output, tuple):
                    grad_output, *_ = grad_output

                if is_input:
                    gradients[name] = grad_input
                else:
                    gradients[name] = grad_output

                if set(names).issubset(gradients.keys()):
                    # HACK: stop the backward pass to save time
                    raise _Finished()

            return forward_hook, backward_hook

        for name, module in model.named_modules():
            if name + ".input" in names:
                forward_hook, backward_hook = make_hooks(name + ".input", True)
                hooks.append(module.register_forward_hook(forward_hook))
                hooks.append(module.register_full_backward_hook(backward_hook))
            if name + ".output" in names:
                forward_hook, backward_hook = make_hooks(name + ".output", False)
                hooks.append(module.register_forward_hook(forward_hook))
                hooks.append(module.register_full_backward_hook(backward_hook))
        with torch.enable_grad():
            try:
                out = model(*args, **kwargs)
                out = output_func(out)
                assert out.ndim == 1, "output_func should reduce to a 1D tensor"
                out.backward(torch.ones_like(out))
            except _Finished:
                pass
    finally:
        # Make sure we always remove hooks even if an exception is raised
        for hook in hooks:
            hook.remove()

        model.zero_grad()

    return activations, gradients
