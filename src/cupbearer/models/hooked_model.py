import contextlib
from collections.abc import Collection

import torch


class HookedModel(torch.nn.Module):
    """A model that can return activations from intermediate layers."""

    def __init__(self):
        super().__init__()
        self._capturing = False
        self._activations: dict[str, torch.Tensor] = {}
        self._names = None

    @contextlib.contextmanager
    def capture(self, names: Collection[str] | None = None):
        self._activations = {}
        self._capturing = True
        self._names = names
        try:
            yield self._activations
        finally:
            self._capturing = False
            self._activations = {}

    def get_activations(self, x, names: Collection[str] | None = None):
        with self.capture(names) as activations:
            out = self(x)
            return out, activations

    def store(self, name: str, value: torch.Tensor):
        if self._capturing and (self._names is None or name in self._names):
            if name in self._activations:
                raise ValueError(f"Activation {name} already stored")
            self._activations[name] = value

    def call_submodule(self, name: str, x):
        """Call a submodule in a way that captures its activations.
        They will be added under their own namespace, starting with '{name}_'
        """
        submodule = getattr(self, name)
        if not self._capturing:
            return submodule(x)

        if not isinstance(submodule, HookedModel):
            raise ValueError(f"{name} does not inherit from ModelHook")
        if self._names is None:
            subnames = None
        else:
            # Get all the names that are in the right namespace, then strip
            # the namespace part to convert them into the right names from the
            # submodule's perspective.
            subnames = {
                n[len(f"{name}_") :] for n in self._names if n.startswith(f"{name}_")
            }
        x, activations = submodule.get_activations(x, subnames)
        for subname, activation in activations.items():
            self.store(f"{name}_{subname}", activation)
        return x
