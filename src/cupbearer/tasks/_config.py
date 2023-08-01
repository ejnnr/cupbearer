from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch.utils.data import Dataset

from cupbearer.data import DatasetConfig
from cupbearer.models import ModelConfig
from cupbearer.models.computations import Model
from cupbearer.utils.utils import BaseConfig


class TaskConfigBase(BaseConfig, ABC):
    @abstractmethod
    def get_reference_data(self) -> Dataset:
        pass

    @abstractmethod
    def get_model(self) -> Model:
        pass

    def get_params(self):
        return None

    @abstractmethod
    def get_anomalous_data(self) -> Dataset:
        pass


# TODO: generalize this so that the fields can be properties, then e.g. adversarial
# examples and backdoors can just inherit from this.
@dataclass(kw_only=True)
class TaskConfig(TaskConfigBase):
    reference_data: DatasetConfig
    model: ModelConfig
    anomalous_data: DatasetConfig

    def get_reference_data(self) -> Dataset:
        return self.reference_data.get_dataset()

    def get_model(self) -> Model:
        return self.model.get_model()

    def get_params(self):
        return self.model.get_params()

    def get_anomalous_data(self) -> Dataset:
        return self.anomalous_data.get_dataset()
