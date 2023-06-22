from abc import ABC, abstractmethod
import functools
from pathlib import Path
import pickle
from typing import Callable, Iterable, Iterator, Protocol, Sized, Union
from torch.utils.data import Dataset


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
        raise ValueError(f"Not a file: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def product(xs: Iterable):
    return functools.reduce(lambda x, y: x * y, xs, 1)
