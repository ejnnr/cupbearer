from abc import ABC
from dataclasses import dataclass
from typing import Optional

from cupbearer.data import (
    DatasetConfig,
    MixedDataConfig,
)
from cupbearer.models import ModelConfig
from cupbearer.models.models import HookedModel


@dataclass(kw_only=True)
class TaskConfig(ABC):
    # Proportion of clean data in untrusted datasets:
    clean_test_weight: float = 0.5
    clean_train_weight: float = 0.5
    # Whether to allow using trusted and untrusted data for training:
    allow_trusted: bool = True
    allow_untrusted: bool = True

    max_train_size: Optional[int] = None
    max_test_size: Optional[int] = None

    def __post_init__(self):
        # We'll only actually instantiate these when we need them, in case relevant
        # attributes get changed after initialization.

        # TODO: I think this is no longer necessary after the config refactor.
        self._trusted_data: Optional[DatasetConfig] = None
        self._untrusted_data: Optional[DatasetConfig] = None
        self._test_data: Optional[MixedDataConfig] = None
        self._model: Optional[ModelConfig] = None

    def _get_clean_data(self, train: bool) -> DatasetConfig:
        raise NotImplementedError

    def _get_anomalous_data(self, train: bool) -> DatasetConfig:
        raise NotImplementedError

    def _get_model(self) -> ModelConfig:
        raise NotImplementedError

    @property
    def trusted_data(self) -> DatasetConfig:
        """Clean data that may be used for training."""
        if not self.allow_trusted:
            raise ValueError(
                "Using trusted training data is not allowed for this task."
            )
        if not self._trusted_data:
            self._trusted_data = self._get_clean_data(train=True)
            self._trusted_data.max_size = self.max_train_size
        return self._trusted_data

    @property
    def untrusted_data(self) -> DatasetConfig:
        """A mix of clean and anomalous data that may be used for training."""
        if not self.allow_untrusted:
            raise ValueError(
                "Using untrusted training data is not allowed for this task."
            )
        if not self._untrusted_data:
            anomalous_data = self._get_anomalous_data(train=True)
            clean_data = self._get_clean_data(train=True)
            self._untrusted_data = MixedDataConfig(
                normal=clean_data,
                anomalous=anomalous_data,
                normal_weight=self.clean_train_weight,
                max_size=self.max_train_size,
                return_anomaly_labels=False,
            )
        return self._untrusted_data

    def build_model(self, input_shape: list[int] | tuple[int]) -> HookedModel:
        if not self._model:
            self._model = self._get_model()
        return self._model.build_model(input_shape)

    @property
    def test_data(self) -> MixedDataConfig:
        if not self._test_data:
            normal = self._get_clean_data(train=False)
            anomalous = self._get_anomalous_data(train=False)
            self._test_data = MixedDataConfig(
                normal=normal,
                anomalous=anomalous,
                normal_weight=self.clean_test_weight,
                max_size=self.max_test_size,
            )
        return self._test_data

    @property
    def num_classes(self):
        try:
            return self.trusted_data.num_classes
        except ValueError:
            return self.untrusted_data.num_classes


@dataclass
class CustomTask(TaskConfig):
    """A fully customizable task config, where all datasets are specified directly."""

    clean_test_data: DatasetConfig
    anomalous_test_data: DatasetConfig
    model: ModelConfig
    clean_train_data: Optional[DatasetConfig] = None
    anomalous_train_data: Optional[DatasetConfig] = None

    def __post_init__(self):
        super(CustomTask, self).__post_init__()
        self.allow_trusted = self.clean_train_data is not None
        self.allow_untrusted = self.anomalous_train_data is not None

    def _get_clean_data(self, train: bool) -> DatasetConfig:
        # This is a bit of a hack because it might return `None`, but that only
        # becomes important if illegal training data is used.
        return self.clean_train_data if train else self.clean_test_data

    def _get_anomalous_data(self, train: bool) -> DatasetConfig:
        # This is a bit of a hack because it might return `None`, but that only
        # becomes important if illegal training data is used.
        return self.anomalous_train_data if train else self.anomalous_test_data

    def _get_model(self) -> ModelConfig:
        return self.model


@dataclass(kw_only=True)
class DebugTaskConfig(TaskConfig):
    """Debug configs for specific tasks can inherit from this for convenience.

    Note that children should inherit this first, to make sure MRO picks up on
    the overriden defaults below!
    """

    # Needs to be at least two because otherwise Mahalanobis distance scores are
    # NaN.
    max_train_size: int = 2
    # Needs to be at least two so it can contain both normal and anomalous data.
    max_test_size: int = 2
