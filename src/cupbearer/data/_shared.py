from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose

from cupbearer.data.transforms import Transform
from cupbearer.utils.scripts import load_config
from cupbearer.utils.utils import BaseConfig


@dataclass(kw_only=True)
class DatasetConfig(BaseConfig, ABC):
    # Only the values of the transforms dict are used, but simple_parsing doesn't
    # support lists of dataclasses, which is why we use a dict. One advantage
    # of this is also that it's easier to override specific transforms.
    # TODO: We should probably make this a list now that we're abandoning CLI.
    transforms: dict[str, Transform] = field(default_factory=dict)
    max_size: Optional[int] = None

    @abstractproperty
    def num_classes(self) -> int:  # type: ignore
        pass

    def get_test_split(self) -> "DatasetConfig":
        # Not every dataset will define this
        raise NotImplementedError

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


@dataclass
class SubsetConfig(DatasetConfig):
    full_dataset: DatasetConfig
    start_fraction: float = 0.0
    end_fraction: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.max_size:
            raise ValueError(
                "max_size should be set on the full dataset, not the subset."
            )
        if self.start_fraction > self.end_fraction:
            raise ValueError(
                f"{self.start_fraction=} must be less than or equal "
                f"to {self.end_fraction=}."
            )
        if self.start_fraction < 0 or self.end_fraction > 1:
            raise ValueError(
                "Fractions must be between 0 and 1, "
                f"got {self.start_fraction} and {self.end_fraction}."
            )
        if self.transforms:
            raise ValueError(
                "Transforms should be applied to the full dataset, not the subset."
            )

    def _build(self) -> Dataset:
        full = self.full_dataset.build()
        start = int(self.start_fraction * len(full))
        end = int(self.end_fraction * len(full))
        return Subset(full, range(start, end))

    @property
    def num_classes(self) -> int:  # type: ignore
        return self.full_dataset.num_classes

    def get_test_split(self) -> "DatasetConfig":
        return SubsetConfig(
            full_dataset=self.full_dataset.get_test_split(),
            start_fraction=self.start_fraction,
            end_fraction=self.end_fraction,
        )

    # Mustn't inherit get_transforms() from full_dataset, they're already applied
    # to the full dataset on build.


def split_dataset_cfg(cfg: DatasetConfig, *fractions: float) -> list[SubsetConfig]:
    if not fractions:
        raise ValueError("At least one fraction must be provided.")
    if not all(0 <= f <= 1 for f in fractions):
        raise ValueError("Fractions must be between 0 and 1.")
    if not sum(fractions) == 1:
        fractions = fractions + (1 - sum(fractions),)

    subsets = []
    current_start = 0.0
    for fraction in fractions:
        subsets.append(SubsetConfig(cfg, current_start, current_start + fraction))
        current_start += fraction
    assert current_start == 1.0
    return subsets


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

    def get_test_split(self) -> DatasetConfig:
        return self.cfg.get_test_split()

    def __post_init__(self):
        self._cfg = None

    @property
    def cfg(self):
        if self._cfg is None:
            # It's important we cache this, not mainly for performance reasons,
            # but because otherwise we'd get different instances every time.
            # Mostly that would be fine, but e.g. the Wanet backdoor transform
            # actually has state not captured by its fields
            # (it's not a "real" dataclass)
            self._cfg = load_config(self.path, "train_data", DatasetConfig)

        return self._cfg

    @property
    def num_classes(self):
        return self.cfg.num_classes

    def _build(self) -> Dataset:
        return self.cfg._build()

    def get_transforms(self) -> list[Transform]:
        transforms = self.cfg.get_transforms() + super().get_transforms()
        return transforms


class MixedData(Dataset):
    def __init__(
        self,
        normal: Dataset,
        anomalous: Dataset,
        normal_weight: float = 0.5,
        return_anomaly_labels: bool = True,
    ):
        self.normal_data = normal
        self.anomalous_data = anomalous
        self.normal_weight = normal_weight
        self.return_anomaly_labels = return_anomaly_labels
        self._length = min(
            int(len(normal) / normal_weight), int(len(anomalous) / (1 - normal_weight))
        )
        self.normal_len = int(self._length * normal_weight)
        self.anomalous_len = self._length - self.normal_len

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < self.normal_len:
            if self.return_anomaly_labels:
                return self.normal_data[index], 0
            return self.normal_data[index]
        else:
            if self.return_anomaly_labels:
                return self.anomalous_data[index - self.normal_len], 1
            return self.anomalous_data[index - self.normal_len]


@dataclass
class MixedDataConfig(DatasetConfig):
    normal: DatasetConfig
    anomalous: DatasetConfig
    normal_weight: float = 0.5
    return_anomaly_labels: bool = True

    def get_test_split(self) -> "MixedDataConfig":
        return MixedDataConfig(
            normal=self.normal.get_test_split(),
            anomalous=self.anomalous.get_test_split(),
            normal_weight=self.normal_weight,
            return_anomaly_labels=self.return_anomaly_labels,
        )

    @property
    def num_classes(self):
        assert (n := self.normal.num_classes) == self.anomalous.num_classes
        return n

    def build(self) -> MixedData:
        # We need to override this method because max_size needs to be applied in a
        # different way: TestDataMix just has normal data first and then anomalous data,
        # if we just used a Subset with indices 1...n, we'd get an incorrect ratio.
        normal = self.normal.build()
        anomalous = self.anomalous.build()
        if self.max_size:
            normal_size = int(self.max_size * self.normal_weight)
            normal_size = min(len(normal), normal_size)
            normal = Subset(normal, range(normal_size))
            anomalous_size = self.max_size - normal_size
            anomalous_size = min(len(anomalous), anomalous_size)
            anomalous = Subset(anomalous, range(anomalous_size))
        dataset = MixedData(
            normal, anomalous, self.normal_weight, self.return_anomaly_labels
        )
        # We don't want to return a TransformDataset here. Transforms should be applied
        # directly to the normal and anomalous data.
        if self.transforms:
            raise ValueError("Transforms are not supported for TestDataConfig.")
        return dataset
