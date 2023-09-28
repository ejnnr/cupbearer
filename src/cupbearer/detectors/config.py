from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from simple_parsing.helpers import mutable_field

from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.models.computations import Model
from cupbearer.utils.scripts import load_config
from cupbearer.utils.utils import BaseConfig, PathConfigMixin


@dataclass
class TrainConfig(BaseConfig):
    pass


@dataclass(kw_only=True)
class DetectorConfig(BaseConfig, ABC):
    train: TrainConfig = mutable_field(TrainConfig)
    max_batch_size: int = 4096

    @abstractmethod
    def build(
        self, model: Model, params, rng, save_dir: Path | None
    ) -> AnomalyDetector:
        pass

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.max_batch_size = 2


@dataclass(kw_only=True)
class StoredDetector(DetectorConfig, PathConfigMixin):
    def build(self, model, params, rng, save_dir) -> AnomalyDetector:
        detector_cfg = load_config(self.get_path(), "detector", DetectorConfig)
        if isinstance(detector_cfg, StoredDetector) and detector_cfg.path == self.path:
            raise RuntimeError(
                f"It looks like the detector you're trying to load from {self.path} "
                "is a stored detector pointing to itself. This probably means "
                "a configuration file is broken."
            )
        detector = detector_cfg.build(model, params, rng, save_dir)
        try:
            detector.load_weights(self.get_path() / "detector")
        except FileNotFoundError:
            logger.warning(
                f"Didn't find weights for detector from {self.path}. "
                "This is normal if the detector doesn't have learned parameters."
            )

        return detector
