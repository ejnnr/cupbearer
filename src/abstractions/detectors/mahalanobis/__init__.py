from dataclasses import dataclass, field

from abstractions.detectors.config import DetectorConfig, TrainConfig

from .mahalanobis_detector import MahalanobisDetector


@dataclass
class MahalanobisTrainConfig(TrainConfig):
    max_batches: int = 0
    relative: bool = False
    rcond: float = 1e-5
    batch_size: int = 4096
    pbar: bool = True


@dataclass
class MahalanobisConfig(DetectorConfig):
    train: MahalanobisTrainConfig = field(default_factory=MahalanobisTrainConfig)

    def build(self, model, params, save_dir) -> MahalanobisDetector:
        return MahalanobisDetector(
            model=model,
            params=params,
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )
