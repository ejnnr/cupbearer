from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from cupbearer.data import DatasetConfig, TestDataConfig, TestDataMix
from cupbearer.models import ModelConfig
from cupbearer.models.computations import Model
from cupbearer.utils.utils import BaseConfig, PathConfigMixin


@dataclass(kw_only=True)
class TaskConfigBase(BaseConfig, ABC, PathConfigMixin):
    @abstractmethod
    def build_train_data(self) -> Dataset:
        pass

    @abstractmethod
    def build_model(self) -> Model:
        pass

    def build_params(self):
        return None

    @abstractmethod
    def build_test_data(self) -> TestDataMix:
        pass


@dataclass(kw_only=True)
class TaskConfig(TaskConfigBase, ABC):
    normal_weight: float = 0.5
    max_train_size: Optional[int] = None
    max_test_size: Optional[int] = None

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            # Needs to be at least two because otherwise Mahalanobis distance scores are
            # NaN.
            self.max_train_size = 2
            # Needs to be at least two so it can contain both normal and anomalous data.
            self.max_test_size = 2

    def __post_init__(self):
        # We'll only actually instantiate these when we need them, in case relevant
        # attributes get changed after initialization.
        self._train_data: Optional[DatasetConfig] = None
        self._test_data: Optional[DatasetConfig] = None
        self._model: Optional[ModelConfig] = None

    @abstractmethod
    def _init_train_data(self):
        pass

    def _get_normal_test_data(self) -> DatasetConfig:
        # Default implementation: just use the training data, but the test split
        # if possible. May be overridden, e.g. if normal test data is meant to be
        # harder or otherwise out-of-distribution.
        if not self._train_data:
            self._init_train_data()
            assert self._train_data is not None, "init_train_data must set _train_data"
        normal = deepcopy(self._train_data)
        if hasattr(normal, "train"):
            # TODO: this is a bit of a hack, maybe there should be a nicer interface
            # for this.
            normal.train = False  # type: ignore

        return normal

    @abstractmethod
    def _get_anomalous_test_data(self) -> DatasetConfig:
        pass

    @abstractmethod
    def _init_model(self):
        pass

    def build_train_data(self) -> Dataset:
        if not self._train_data:
            self._init_train_data()
            assert self._train_data is not None, "init_train_data must set _train_data"
            self._train_data.max_size = self.max_train_size
        return self._train_data.build()

    def build_model(self) -> Model:
        if not self._model:
            self._init_model()
            assert self._model is not None, "init_model must set _model"
        return self._model.build_model()

    def build_params(self):
        if not self._model:
            self._init_model()
            assert self._model is not None, "init_model must set _model"
        return self._model.build_params()

    def build_test_data(self) -> TestDataMix:
        normal = self._get_normal_test_data()
        anomalous = self._get_anomalous_test_data()
        self._test_data = TestDataConfig(
            normal=normal,
            anomalous=anomalous,
            normal_weight=self.normal_weight,
            max_size=self.max_test_size,
        )
        return self._test_data.build()
