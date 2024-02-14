from dataclasses import dataclass, field

from cupbearer.detectors.config import ActivationBasedDetectorConfig
from cupbearer.utils.utils import BaseConfig

from .mahalanobis_detector import MahalanobisDetector


@dataclass
class MahalanobisTrainConfig(BaseConfig):
    max_batches: int = 0
    relative: bool = False
    rcond: float = 1e-5
    batch_size: int = 4096
    max_batch_size: int = 4096
    pbar: bool = True
    debug: bool = False


@dataclass
class DebugMahalanobisTrainConfig(MahalanobisTrainConfig):
    max_batches: int = 2
    batch_size: int = 2
    max_batch_size: int = 2


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


@dataclass
class DebugMahalanobisConfig(MahalanobisConfig):
    train: DebugMahalanobisTrainConfig = field(
        default_factory=DebugMahalanobisTrainConfig
    )
