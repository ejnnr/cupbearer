from dataclasses import dataclass, field

from cupbearer.detectors.config import DetectorConfig, TrainConfig

from .mahalanobis_detector import MahalanobisDetector


@dataclass
class MahalanobisTrainConfig(TrainConfig):
    max_batches: int = 0
    relative: bool = False
    rcond: float = 1e-5
    batch_size: int = 4096
    pbar: bool = True
    debug: bool = False

    def _set_debug(self):
        self.max_batches = 2
        self.batch_size = 2


@dataclass
class MahalanobisConfig(DetectorConfig):
    train: MahalanobisTrainConfig = field(default_factory=MahalanobisTrainConfig)

    def build(self, model, params, rng, save_dir) -> MahalanobisDetector:
        return MahalanobisDetector(
            model=model,
            params=params,
            rng=rng,
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )
