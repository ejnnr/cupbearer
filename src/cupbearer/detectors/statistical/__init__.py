from abc import ABC
from dataclasses import dataclass, field

from cupbearer.detectors.config import ActivationBasedDetectorConfig
from cupbearer.utils.utils import BaseConfig

from .mahalanobis_detector import MahalanobisDetector


@dataclass
class StatisticalTrainConfig(BaseConfig, ABC):
    max_batches: int = 0
    batch_size: int = 4096
    max_batch_size: int = 4096
    pbar: bool = True
    debug: bool = False

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.batch_size = 2
            self.max_batch_size = 2


@dataclass
class MahalanobisTrainConfig(StatisticalTrainConfig):
    relative: bool = False
    rcond: float = 1e-5


@dataclass
class MahalanobisConfig(ActivationBasedDetectorConfig):
    train: MahalanobisTrainConfig = field(default_factory=MahalanobisTrainConfig)

    def build(self, model, save_dir) -> MahalanobisDetector:
        return MahalanobisDetector(
            model=model,
            activation_name_func=self.resolve_name_func(),
            max_batch_size=self.train.max_batch_size,
            save_path=save_dir,
        )
