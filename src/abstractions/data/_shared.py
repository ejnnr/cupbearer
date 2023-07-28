import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose

from abstractions.utils.scripts import load_config
from abstractions.utils.utils import BaseConfig


@dataclass
class Transform(BaseConfig, ABC):
    @abstractmethod
    def __call__(self, sample):
        pass


@dataclass(kw_only=True)
class DatasetConfig(BaseConfig, ABC):
    # Only the values of the transforms dict are used, but simple_parsing doesn't
    # support lists of dataclasses, which is why we use a dict. One advantage
    # of this is also that it's easier to override specific transforms.
    transforms: dict[str, Transform] = field(default_factory=dict)
    max_size: Optional[int] = None

    def get_dataset(self) -> Dataset:
        """Create an instance of the Dataset described by this config."""
        dataset = self._get_dataset()
        transform = Compose(list(self.transforms.values()))
        # add_transforms(dataset, transform)
        dataset = TransformDataset(dataset, transform)
        if self.max_size:
            dataset = Subset(dataset, range(self.max_size))
        return dataset

    @abstractmethod
    def _get_dataset(self) -> Dataset:
        pass

    def _set_debug(self):
        super()._set_debug()
        self.max_size = 2


def numpy_collate(batch):
    """Variant of the default collate_fn that returns ndarrays instead of tensors."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return np.array(batch)


def adapt_transform(transform):
    """Adapt a transform designed to work on inputs to work on entire samples."""

    @functools.wraps(transform)
    def adapted(sample):
        if isinstance(sample, tuple):
            img, *rest = sample
        else:
            img = sample
            rest = None

        img = transform(img)

        if rest is None:
            return img
        return (img, *rest)

    return adapted


# Needs to be a dataclass to make simple_parsing's serialization work correctly.
@dataclass
class ToNumpy(Transform):
    def __call__(self, sample):
        return _to_numpy(sample)


@adapt_transform
def _to_numpy(img):
    out = np.array(img, dtype=jnp.float32) / 255.0
    if out.ndim == 2:
        # Add a channel dimension. Note that flax.linen.Conv expects
        # the channel dimension to be the last one.
        out = np.expand_dims(out, axis=-1)
    return out


class TransformDataset(Dataset):
    """Dataset that applies a transform to another dataset."""

    def __init__(self, dataset: Dataset, transform: Transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.transform(sample)


@dataclass
class TrainDataFromRun(DatasetConfig):
    path: Path

    def _get_dataset(self) -> Dataset:
        data_cfg = load_config(self.path, "train_data", DatasetConfig)
        return data_cfg.get_dataset()
