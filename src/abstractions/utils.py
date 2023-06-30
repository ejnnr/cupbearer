from abc import ABC, abstractmethod
import functools
from pathlib import Path
import pickle
import re
import shutil
from typing import Callable, Iterable, Iterator, Protocol, Sized, Union
from hydra import TaskFunction
from loguru import logger
from torch.utils.data import Dataset
import os
from hydra.utils import to_absolute_path, get_original_cwd
from hydra.experimental.callback import Callback
from typing import Any
from omegaconf import DictConfig


class SizedIterable(Protocol):
    # Apparently there's no type in the standard library for this.
    # Collection also requires __contains__, which pytorch Dataloaders in general
    # don't implement.

    def __iter__(self) -> Iterator:
        ...

    def __len__(self) -> int:
        ...


def add_transforms(cls):
    """
    A decorator for PyTorch Dataset classes that adds a `transforms` argument to the constructor.
    The `transforms` argument is a single function that takes multiple values (such as image and target) as input
    and returns transformed versions of them all. This is useful when we want to apply random transforms
    to multiple parts of a sample that aren't independent, e.g. transform the label with the image.

    Args:
        cls (type): A class that inherits from torch.utils.data.Dataset.

    Returns:
        type: A new class that inherits from the input class and has the additional `transforms` functionality.

    Raises:
        TypeError: If the input class is not a subclass of torch.utils.data.Dataset.
    """
    if not issubclass(cls, Dataset):
        raise TypeError(f"{cls} must be a subclass of torch.utils.data.Dataset")

    assert issubclass(cls, Sized)  # Tell pylance that Datasets have len()

    @functools.wraps(cls, updated=())
    class TransformedDataset(cls):
        def __init__(self, *args, transforms=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._transforms = transforms

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            if self._transforms:
                sample = self._transforms(sample)
            return sample

    return TransformedDataset


def adapt_transform(transform: Callable) -> Callable:
    """
    A decorator that takes a torchvision transform and turns it into a new one that can be used for the
    `transforms` argument in a Dataset class created with the `add_transforms` decorator. The new transform
    keeps all but the first value unchanged. The decorator works on both functions and callable classes.

    Args:
        transform (callable): A torchvision transform function or callable class.

    Returns:
        callable: A new transform function or class that takes an image and other values as input
                  and returns the transformed image and the unchanged remaining values.

    Raises:
        TypeError: If the input transform is not a function or class.
    """
    if not callable(transform):
        raise TypeError(f"{transform} must be a function or class")

    if isinstance(transform, type):

        @functools.wraps(transform, updated=())
        class AdaptedTransform(transform):
            def __call__(self, sample):
                img, *args = sample
                return super().__call__(img), *args

        return AdaptedTransform

    else:

        @functools.wraps(transform)
        def adapted_transform(sample):
            img, *args = sample
            return transform(img), *args

        return adapted_transform


SUFFIX = ".pytree"


# Based on https://github.com/google/jax/issues/2116#issuecomment-580322624
# TODO: basically just using this for nested dicts of arrays, so a generalized version
# of jnp.savez should work and be faster than pickle.
def save(data, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f"File {path} already exists.")
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load(path: Union[str, Path]):
    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def product(xs: Iterable):
    return functools.reduce(lambda x, y: x * y, xs, 1)


def original_relative_path(path: str | Path) -> Path:
    """Converts a path to be relative to the original (pre-hydra) working directory."""
    new_cwd = os.getcwd()
    abs_path = Path(new_cwd) / Path(path)
    original_cwd = get_original_cwd()
    rel_path = os.path.relpath(abs_path, original_cwd)
    return Path(rel_path)


class CheckOutputDirExistsCallback(Callback):
    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        # This hook isn't really necessary given that on_job_start below takes
        # care of similar things. But on non-multiruns, this hook will be called
        # *before* the .hydra dir is overwritten, so it gives us more safety at
        # least for those cases.
        if os.path.exists(config.hydra.run.dir):
            if config.get("overwrite_output", False):
                logger.info("Overwriting output dir")
                shutil.rmtree(config.hydra.run.dir)
            else:
                raise BaseException(
                    "Output dir already exists! Use +overwrite_output=true to overwrite."
                )

    def on_job_start(
        self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any
    ):
        """Check that the output dir is empty, except for the .hydra dir and the logfile.

        This hook is needed for multirun jobs: the previous one won't trigger, but we don't
        want to use on_multirun_start because that would only allow checking whether the
        entire sweep directory exists. Instead, we'd like overwrite checks on a job-basis.

        TODO: The downside of this approach is that this hook is only called after the
        .hydra dir has already been created and potentially overwrote the existing one.
        So there's a danger of data loss even with overwrite_output=false.
        """
        path = Path(config.hydra.runtime.output_dir)
        files = os.listdir(path)
        if not set(files) <= {".hydra", f"{config.hydra.job.name}.log"}:
            if config.get("overwrite_output", False):
                logger.info("Overwriting output dir")
                # Don't remove the .hydra dir, that's from the new run and already
                # overwrites the old one anyway.
                # Do delete the log file because otherwise hydra will append to it.
                # (There shouldn't be anything in the logfile yet.)
                files.remove(".hydra")
                for file in files:
                    os.remove(path / file)
            else:
                raise BaseException(
                    "Output dir already exists! Use +overwrite_output=true to overwrite. "
                    f".hydra dir at {path / '.hydra'} has already been overwritten!"
                )


# https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
def escape_ansi(line):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)
