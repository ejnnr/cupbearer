import codecs
import importlib
import pickle
from pathlib import Path
from typing import Union

import torch

SUFFIX = ".pt"
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
    if isinstance(leaf, (str, int, float, bool, torch.Tensor)):
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


def tree_map(f, tree):
    """Like jax.tree_map, but simpler and for pytorch."""
    # We could use https://github.com/metaopt/optree in the future,
    # which would be faster and generally add support for various tree operations.
    if isinstance(tree, list):
        return [tree_map(f, x) for x in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(f, x) for x in tree)
    if isinstance(tree, dict):
        return {k: tree_map(f, v) for k, v in tree.items()}
    try:
        return f(tree)
    except Exception as e:
        raise ValueError(
            f"Could not apply {f} to leaf {tree} of type {type(tree)}"
        ) from e


def save(data, path: Union[str, Path], overwrite: bool = False):
    data = tree_map(validate_and_convert_leaf, data)
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            assert not path.is_dir(), f"{path} is a directory, won't overwrite"
            path.unlink()
        else:
            raise RuntimeError(f"File {path} already exists.")
    torch.save(data, path.with_suffix(SUFFIX))


def load(path: Union[str, Path]):
    path = Path(path)
    if path.is_dir():
        raise ValueError(
            f"Expected a file, got directory {path}. "
            "Maybe this is in the legacy Jax format?"
        )

    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
    with open(path, "rb") as file:
        data = torch.load(file)
        data = tree_map(from_string, data)
        return data


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


def inputs_from_batch(batch):
    # batch may contain labels or other info, if so we strip it out
    if isinstance(batch, (tuple, list)):
        return batch[0]
    else:
        return batch
