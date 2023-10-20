from abc import ABC, abstractmethod, abstractproperty
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

    def store(self, basepath):
        """Save transform state to reproduce instance later."""
        pass

    def load(self, basepath):
        """Load transform state to reproduce stored instance."""
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

    @abstractproperty
    def num_classes(self) -> int:
        pass

    def get_transforms(self) -> list[Transform]:
        """Return a list of transforms that should be applied to this dataset.

        Most subclasses won't need to override this, since it just returns
        the transforms field by default. But in some cases, we need to apply custom
        processing to this that can't be handled in __post_init__ (see BackdoorData
        for an example).
        """
        return list(self.transforms.values())

    def build(self) -> Dataset:
        """Create an instance of the Dataset described by this config."""
        dataset = self._build()
        transform = Compose(self.get_transforms())
        dataset = TransformDataset(dataset, transform)
        if self.max_size:
            assert self.max_size <= len(dataset)
            dataset = Subset(dataset, range(self.max_size))
        return dataset

    def _build(self) -> Dataset:
        # Not an abstractmethod because e.g. TestDataConfig overrides build() instead.
        raise NotImplementedError

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
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

    def __post_init__(self):
        self._cfg = None

    @property
    def cfg(self):
        if self._cfg is None:
            # It's important we cache this, not mainly for performance reasons,
            # but because otherwise we'd get different instances every time.
            # Mostly that would be fine, but e.g. the Wanet backdoor transform
            # actually has state not captures by its field (it's not a "real" dataclass)
            self._cfg = load_config(self.path, "train_data", DatasetConfig)

        return self._cfg

    @property
    def num_classes(self):
        return self.cfg.num_classes

    def _build(self) -> Dataset:
        return self.cfg._build()

    def get_transforms(self) -> list[Transform]:
        transforms = self.cfg.get_transforms() + super().get_transforms()
        print(transforms)
        return transforms


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

    @property
    def num_classes(self):
        assert (n := self.normal.num_classes) == self.anomalous.num_classes
        return n

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


@dataclass
class NoData(DatasetConfig):
    """Dummy class for non-existent datasets.

    This is a workaround because simple_parsing doesn't support None as an option
    when using subgroups, so we can pass this instead when we mean "no dataset".

    It still needs to be instantiable, but otherwise should raise NotImplementedError
    on any operation.
    """

    @property
    def num_classes(self):
        raise NotImplementedError

    # build already raises NotImplementedError by default
