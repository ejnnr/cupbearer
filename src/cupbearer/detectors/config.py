from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from simple_parsing.helpers import mutable_field

from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.models.computations import Model
from cupbearer.utils.scripts import load_config
from cupbearer.utils.utils import BaseConfig


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

    def _set_debug(self):
        super()._set_debug()
        self.max_batch_size = 2


@dataclass(kw_only=True)
class StoredDetector(DetectorConfig):
    # TODO: It might be nice to just use save_dir for this when calling build(),
    # since it's usually going to be the same as path. But then this doesn't get
    # stored in the config file, which breaks loading detectors.
    path: Path

    def build(self, model, params, rng, save_dir) -> AnomalyDetector:
        detector_cfg = load_config(self.path, "detector", DetectorConfig)
        if isinstance(detector_cfg, StoredDetector) and detector_cfg.path == self.path:
            raise RuntimeError(
                f"It looks like the detector you're trying to load from {self.path} "
                "is a stored detector pointing to itself. This probably means "
                "a configuration file is broken."
            )
        detector = detector_cfg.build(model, params, rng, save_dir)
        try:
            detector.load_weights(self.path / "detector")
        except FileNotFoundError:
            pass

        return detector
