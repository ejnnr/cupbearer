from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from simple_parsing.helpers import mutable_field

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


# TODO: this feels like unnecessary indirection, can maybe integrate this elsewhere
@dataclass(kw_only=True)
class ActivationBasedDetectorConfig(DetectorConfig):
    name_func: Optional[str] = None

    def resolve_name_func(self) -> Callable[[HookedModel], Collection[str]] | None:
        if isinstance(self.name_func, str):
            return get_object(self.name_func)
        return self.name_func


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
