from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from cupbearer.data import (
    DatasetConfig,
    MixedDataConfig,
    split_dataset_cfg,
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

    def _get_trusted_data(self) -> DatasetConfig:
        raise NotImplementedError

    def _get_clean_untrusted_data(self) -> DatasetConfig:
        raise NotImplementedError

    def _get_anomalous_data(self) -> DatasetConfig:
        raise NotImplementedError

    # The following two methods don't need to be implemented, the task will use
    # get_test_split() on the untrusted data by default.
    def _get_clean_test_data(self) -> DatasetConfig:
        raise NotImplementedError

    def _get_anomalous_test_data(self) -> DatasetConfig:
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
            self._trusted_data = deepcopy(self._get_trusted_data())
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
            anomalous_data = self._get_anomalous_data()
            clean_data = self._get_clean_untrusted_data()
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
            try:
                anomalous_data = self._get_anomalous_test_data()
                clean_data = self._get_clean_test_data()
            except NotImplementedError:
                anomalous_data = self._get_anomalous_data().get_test_split()
                clean_data = self._get_clean_untrusted_data().get_test_split()
            self._test_data = MixedDataConfig(
                normal=clean_data,
                anomalous=anomalous_data,
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
class FuzzedTask(TaskConfig):
    """A task where the anomalous inputs are some modified version of clean ones."""

    trusted_fraction: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        # First we get the base (unmodified) data and its test split.
        train_data = self._get_base_data()
        test_data = train_data.get_test_split()

        # We split the training data up into three parts:
        # 1. A `trusted_fraction` part will be used as trusted data.
        # 2. Out of the remaining part, a `clean_untrusted_fraction` part will be used
        #    as clean untrusted data.
        # 3. The rest will be used as anomalous training data.
        (
            self._trusted_data,
            self._clean_untrusted_data,
            _anomalous_base,
        ) = split_dataset_cfg(
            train_data,
            self.trusted_fraction,
            # Using clean_train_weight here means we'll end up using all our data,
            # since this is also what's used later in the MixedDataConfig.
            (1 - self.trusted_fraction) * self.clean_train_weight,
            (1 - self.trusted_fraction) * (1 - self.clean_train_weight),
        )

        # Similarly, we plit up the test data, except there is no trusted subset.
        self._clean_test_data, _anomalous_test_base = split_dataset_cfg(
            test_data,
            self.clean_test_weight,
        )

        self._anomalous_data = self.fuzz(_anomalous_base)
        self._anomalous_test_data = self.fuzz(_anomalous_test_base)

    @abstractmethod
    def fuzz(self, data: DatasetConfig) -> DatasetConfig:
        pass

    @abstractmethod
    def _get_base_data(self) -> DatasetConfig:
        pass

    def _get_trusted_data(self) -> DatasetConfig:
        return self._trusted_data

    def _get_clean_untrusted_data(self) -> DatasetConfig:
        return self._clean_untrusted_data

    def _get_anomalous_data(self) -> DatasetConfig:
        return self._anomalous_data

    def _get_clean_test_data(self) -> DatasetConfig:
        return self._clean_test_data

    def _get_anomalous_test_data(self) -> DatasetConfig:
        return self._anomalous_test_data


@dataclass(kw_only=True)
class CustomTask(TaskConfig):
    """A fully customizable task config, where all datasets are specified directly."""

    trusted_data: DatasetConfig
    clean_untrusted_data: DatasetConfig
    anomalous_data: DatasetConfig
    model: ModelConfig

    def _get_clean_untrusted_data(self) -> DatasetConfig:
        return self.clean_untrusted_data

    def _get_trusted_data(self) -> DatasetConfig:
        return self.trusted_data

    def _get_anomalous_data(self) -> DatasetConfig:
        return self.anomalous_data

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
