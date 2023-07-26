import copy
import dataclasses
import functools
import orbax.checkpoint
import inspect
import pickle
import shutil
from pathlib import Path
from typing import Iterable, TypeVar, Union
import flax
from flax.core import FrozenDict

import jax


SUFFIX = ".pytree"


def save(data, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            assert path.is_dir(), f"{path} is not a directory, won't overwrite"
            shutil.rmtree(path)
        else:
            raise RuntimeError(f"File {path} already exists.")
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # If leaves are strings, these leaves must be saved with aggregate=True.
    # TODO: I'm guessing we should not use this for arrays, given that it's not the
    # default?
    # But if we care about performance, should probably just use the new OCDBT version,
    # see https://orbax.readthedocs.io/en/latest/optimized_checkpointing.html
    save_args = jax.tree_map(lambda _: orbax.checkpoint.SaveArgs(aggregate=True), data)
    checkpointer.save(path, data, save_args=save_args)


def load(path: Union[str, Path]):
    path = Path(path)
    if path.is_dir():
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        data = checkpointer.restore(path)
        return data

    # Support for legacy pickle format:
    pickle_path = path
    if path.suffix != SUFFIX:
        pickle_path = path.with_suffix(SUFFIX)
    if pickle_path.is_file():
        with open(pickle_path, "rb") as file:
            return pickle.load(file)

    raise FileNotFoundError(f"Found neither {path} nor {pickle_path}")


def product(xs: Iterable):
    return functools.reduce(lambda x, y: x * y, xs, 1)


def negative(tree):
    return jax.tree_map(lambda x: -x, tree)


def weighted_sum(tree1, tree2, alpha):
    return jax.tree_map(lambda x, y: alpha * x + (1 - alpha) * y, tree1, tree2)


def merge_dicts(a: dict | FrozenDict, b: dict | FrozenDict) -> dict | FrozenDict:
    """Merges two dictionaries recursively."""

    if isinstance(a, FrozenDict):
        frozen = True
        merged = a.unfreeze()
    else:
        frozen = False
        merged = a.copy()
    for key, value in b.items():
        if key in merged and isinstance(merged[key], (dict, FrozenDict)):
            # Make sure we don't overwrite a dict with a non-dict
            assert isinstance(value, (dict, FrozenDict))
            merged[key] = merge_dicts(merged[key], value)
        else:
            if isinstance(value, (dict, FrozenDict)):
                # Make sure we don't overwrite a non-dict with a dict
                assert key not in merged
            merged[key] = value

    if frozen:
        merged = flax.core.freeze(merged)
    return merged


def store_init_args(cls, ignore={"self"}):
    orig_init = cls.__init__
    sig = inspect.signature(orig_init)

    @functools.wraps(cls.__init__)
    def new_init(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        self._init_kwargs = dict(bound.arguments)
        for name in ignore:
            self._init_kwargs.pop(name, None)
        orig_init(self, *args, **kwargs)

    cls.__init__ = new_init

    @property
    def init_kwargs_property(self):
        return self._init_kwargs

    cls.init_kwargs = init_kwargs_property
    return cls


storable = functools.partial(store_init_args, ignore={"self", "model", "params"})

T = TypeVar("T")


def mutable_field(default: T = None) -> T:
    return dataclasses.field(default_factory=lambda: copy.deepcopy(default))


def list_field():
    return dataclasses.field(default_factory=list)


def dict_field():
    return dataclasses.field(default_factory=dict)
