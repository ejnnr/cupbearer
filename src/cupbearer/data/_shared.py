from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode, resize

from cupbearer.utils.scripts import load_config
from cupbearer.utils.utils import BaseConfig


@dataclass
class Transform(BaseConfig, ABC):
    @abstractmethod
    def __call__(self, sample):
        pass


@dataclass
class AdaptedTransform(Transform, ABC):
    """Adapt a transform designed to work on inputs to work on img, label pairs."""

    @abstractmethod
    def __img_call__(self, img):
        pass

    def __rest_call__(self, *rest):
        return (*rest,)

    def __call__(self, sample):
        if isinstance(sample, tuple):
            img, *rest = sample
        else:
            img = sample
            rest = None

        img = self.__img_call__(img)

        if rest is None:
            return img
        else:
            rest = self.__rest_call__(*rest)

        return (img, *rest)


@dataclass(kw_only=True)
class DatasetConfig(BaseConfig, ABC):
    # Only the values of the transforms dict are used, but simple_parsing doesn't
    # support lists of dataclasses, which is why we use a dict. One advantage
    # of this is also that it's easier to override specific transforms.
    transforms: dict[str, Transform] = field(default_factory=dict)
    max_size: Optional[int] = None

    def build(self) -> Dataset:
        """Create an instance of the Dataset described by this config."""
        dataset = self._build()
        transform = Compose(list(self.transforms.values()))
        dataset = TransformDataset(dataset, transform)
        if self.max_size:
            assert self.max_size <= len(dataset)
            dataset = Subset(dataset, range(self.max_size))
        return dataset

    def _build(self) -> Dataset:
        # Not an abstractmethod because e.g. TestDataConfig overrides build() instead.
        raise NotImplementedError

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


# Needs to be a dataclass to make simple_parsing's serialization work correctly.
@dataclass
class ToNumpy(AdaptedTransform):
    def __img_call__(self, img):
        out = np.array(img, dtype=jnp.float32) / 255.0
        if out.ndim == 2:
            # Add a channel dimension. Note that flax.linen.Conv expects
            # the channel dimension to be the last one.
            out = np.expand_dims(out, axis=-1)
        return out


@dataclass
class Resize(AdaptedTransform):
    size: tuple[int, ...]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR
    max_size: Optional[int] = None
    antialias: Optional[Union[str, bool]] = "warn"

    def __img_call__(self, img):
        return resize(
            img,
            size=self.size,
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )


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

    def _build(self) -> Dataset:
        data_cfg = load_config(self.path, "train_data", DatasetConfig)
        return data_cfg.build()


class TestDataMix(Dataset):
    def __init__(
        self,
        normal: Dataset,
        anomalous: Dataset,
        normal_weight: float = 0.5,
    ):
        self.normal_data = normal
        self.anomalous_data = anomalous
        self.normal_weight = normal_weight
        self._length = min(
            int(len(normal) / normal_weight), int(len(anomalous) / (1 - normal_weight))
        )
        self.normal_len = int(self._length * normal_weight)
        self.anomalous_len = self._length - self.normal_len

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < self.normal_len:
            return self.normal_data[index], 0
        else:
            return self.anomalous_data[index - self.normal_len], 1


@dataclass
class TestDataConfig(DatasetConfig):
    normal: DatasetConfig
    anomalous: DatasetConfig
    normal_weight: float = 0.5

    def build(self) -> TestDataMix:
        # We need to override this method because max_size needs to be applied in a
        # different way: TestDataMix just has normal data first and then anomalous data,
        # if we just used a Subset with indices 1...n, we'd get an incorrect ratio.
        normal = self.normal.build()
        anomalous = self.anomalous.build()
        if self.max_size:
            normal_size = int(self.max_size * self.normal_weight)
            assert normal_size <= len(normal)
            normal = Subset(normal, range(normal_size))
            anomalous_size = self.max_size - normal_size
            assert anomalous_size <= len(anomalous)
            anomalous = Subset(anomalous, range(anomalous_size))
        dataset = TestDataMix(normal, anomalous, self.normal_weight)
        # We don't want to return a TransformDataset here. Transforms should be applied
        # directly to the normal and anomalous data.
        if self.transforms:
            raise ValueError("Transforms are not supported for TestDataConfig.")
        return dataset
