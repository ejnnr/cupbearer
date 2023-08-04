import codecs
import copy
import dataclasses
import functools
import importlib
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar, Union

import flax
import jax
import numpy as np
import orbax.checkpoint
from flax.core import FrozenDict
from simple_parsing.helpers import serialization

SUFFIX = ".pytree"
TYPE_PREFIX = "__TYPE__:"
PICKLE_PREFIX = "__PICKLE__:"


def from_string(s):
    # Doesn't restore Paths but all the code should be able to handle getting strings
    # instead.
    if not isinstance(s, str):
        return s
    if s.startswith(TYPE_PREFIX):
        s = s[len(TYPE_PREFIX) :]
        return get_object(s)
    if s.startswith(PICKLE_PREFIX):
        s = s[len(PICKLE_PREFIX) :]
        pickled = codecs.decode(s.encode(), "base64")
        return pickle.loads(pickled)

    return s


def validate_and_convert_leaf(leaf):
    if isinstance(leaf, (str, int, float, bool, jax.Array, np.ndarray)):
        return leaf
    if isinstance(leaf, Path):
        return str(leaf)
    if isinstance(leaf, type):
        return TYPE_PREFIX + leaf.__module__ + "." + leaf.__name__

    try:
        pickled = pickle.dumps(leaf)
    except Exception as e:
        raise ValueError(f"Could not pickle object {leaf}") from e
    # Make sure we're not accidentally encoding huge objects inefficiently into strings:
    if len(pickled) > 1e6:
        raise ValueError(
            f"Object of type {type(leaf)} has {round(len(pickled) / 1e6, 1)} MB "
            "when pickled. This is probably a mistake."
        )
    pickle_str = codecs.encode(pickled, "base64").decode()
    return PICKLE_PREFIX + pickle_str


def save(data, path: Union[str, Path], overwrite: bool = False):
    data = jax.tree_map(validate_and_convert_leaf, data)
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
        data = jax.tree_map(from_string, data)
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


T = TypeVar("T")


def mutable_field(default: T = None) -> T:
    return dataclasses.field(default_factory=lambda: copy.deepcopy(default))


def list_field():
    return dataclasses.field(default_factory=list)


def dict_field():
    return dataclasses.field(default_factory=dict)


# https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
def escape_ansi(line):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


@dataclass(kw_only=True)
class BaseConfig(serialization.serializable.Serializable):
    debug: bool = False

    def to_dict(
        self,
        dict_factory: type[dict] = dict,
        recurse: bool = True,
        save_dc_types: bool = True,
    ) -> dict:
        # This is the only change we make: default is for save_dc_types to be False.
        # Instead, we always pass `True`. (We don't want the default elsewhere
        # to get passed here and override this.)
        # We could pass save_dc_types to `save`, but that doesn't propagate into
        # lists of dataclasses.
        return serialization.serializable.to_dict(
            self, dict_factory, recurse, save_dc_types=True
        )

    def __post_init__(self):
        if self.debug:
            self._set_debug()

    def _set_debug(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, BaseConfig):
                value.debug = True
                value._set_debug()


def get_object(path: str):
    """Get an object from a string.

    Args:
      path: A string of the form "module.submodule.object_name".

    Returns:
      The object named by `path`.
    """
    module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)
