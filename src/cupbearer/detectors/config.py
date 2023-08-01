from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from simple_parsing.helpers import mutable_field

from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.models.computations import Model
from cupbearer.utils.utils import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    pass


@dataclass(kw_only=True)
class DetectorConfig(BaseConfig, ABC):
    train: TrainConfig = mutable_field(TrainConfig)
    max_batch_size: int = 4096

    @abstractmethod
    def build(self, model: Model, params, save_dir: Path | None) -> AnomalyDetector:
        pass

    def _set_debug(self):
        super()._set_debug()
        self.max_batch_size = 2


@dataclass(kw_only=True)
class StoredDetector(DetectorConfig):
    def build(self, model, params, save_dir) -> AnomalyDetector:
        if save_dir is None:
            raise ValueError("Must specify directory when using StoredDetector")
        return AnomalyDetector.load(save_dir / "detector", model, params)
