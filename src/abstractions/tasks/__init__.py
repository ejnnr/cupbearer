from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

from abstractions.data import DatasetConfig
from abstractions.models import ModelConfig
from abstractions.models.computations import Model
from abstractions.utils.hydra import hydra_config, hydra_config_base


@hydra_config_base("task")  # type: ignore
class TaskConfigBase(ABC):
    @abstractmethod
    def get_reference_data(self) -> Dataset:
        pass

    @abstractmethod
    def get_model(self) -> Model:
        pass

    @abstractmethod
    def get_anomalous_data(self) -> Dataset:
        pass


@hydra_config
@dataclass
class TaskConfig:
    reference_data: DatasetConfig
    model: ModelConfig
    anomalous_data: DatasetConfig

    def get_reference_data(self) -> Dataset:
        return self.reference_data.get_dataset()

    def get_model(self) -> Model:
        return self.model.get_model()

    def get_anomalous_data(self) -> Dataset:
        return self.anomalous_data.get_dataset()
