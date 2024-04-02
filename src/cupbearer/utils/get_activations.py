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
                    activations[name] = (
                        input if isinstance(input, torch.Tensor) else input[0]
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
