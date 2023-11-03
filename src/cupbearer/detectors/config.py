from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from simple_parsing.helpers import mutable_field

from cupbearer import models
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.models.models import HookedModel
from cupbearer.utils.scripts import load_config
from cupbearer.utils.utils import BaseConfig, PathConfigMixin, get_object


@dataclass
class TrainConfig(BaseConfig):
    pass


@dataclass(kw_only=True)
class DetectorConfig(BaseConfig, ABC):
    train: TrainConfig = mutable_field(TrainConfig)
    max_batch_size: int = 4096

    @abstractmethod
    def build(self, model: HookedModel, save_dir: Path | None) -> AnomalyDetector:
        pass

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.max_batch_size = 2


@dataclass(kw_only=True)
class ActivationBasedDetectorConfig(DetectorConfig):
    names: str = "cupbearer.detectors.config.get_default_names"

    def get_names(self, model: HookedModel):
        name_func = get_object(self.names)
        return name_func(model)


def get_default_names(model: HookedModel):
    if isinstance(model, models.MLP):
        return [f"post_linear_{i}" for i in range(len(model.layers))]
    elif isinstance(model, models.CNN):
        return [f"post_conv_{i}" for i in range(len(model.conv_layers))] + [
            f"post_linear_{i}" for i in range(len(model.mlp.layers))
        ]
    else:
        raise ValueError(f"Unknown model type {type(model)}")


@dataclass(kw_only=True)
class StoredDetector(DetectorConfig, PathConfigMixin):
    def build(self, model, save_dir) -> AnomalyDetector:
        detector_cfg = load_config(self.get_path(), "detector", DetectorConfig)
        if isinstance(detector_cfg, StoredDetector) and detector_cfg.path == self.path:
            raise RuntimeError(
                f"It looks like the detector you're trying to load from {self.path} "
                "is a stored detector pointing to itself. This probably means "
                "a configuration file is broken."
            )
        detector = detector_cfg.build(model, save_dir)
        try:
            detector.load_weights(self.get_path() / "detector")
        except FileNotFoundError:
            logger.warning(
                f"Didn't find weights for detector from {self.path}. "
                "This is normal if the detector doesn't have learned parameters."
            )

        return detector
