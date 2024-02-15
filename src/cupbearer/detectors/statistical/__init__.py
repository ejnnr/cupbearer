from dataclasses import dataclass, field

from cupbearer.detectors.config import ActivationBasedDetectorConfig

from .mahalanobis_detector import MahalanobisDetector
from .que_detector import QuantumEntropyDetector
from .spectral_detector import SpectralSignatureDetector
from .statistical import (
    ActivationCovarianceTrainConfig,
    MahalanobisTrainConfig,
)


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
class SpectralSignatureConfig(ActivationBasedDetectorConfig):
    train: ActivationCovarianceTrainConfig = field(
        default_factory=ActivationCovarianceTrainConfig
    )
    train_on_clean: bool = False

    def build(self, model, save_dir) -> SpectralSignatureDetector:
        return SpectralSignatureDetector(
            model=model,
            activation_name_func=self.resolve_name_func(),
            max_batch_size=self.train.max_batch_size,
            save_path=save_dir,
        )


@dataclass
class QuantumEntropyConfig(ActivationBasedDetectorConfig):
    train: ActivationCovarianceTrainConfig = field(
        default_factory=ActivationCovarianceTrainConfig
    )

    def build(self, model, save_dir) -> QuantumEntropyDetector:
        return QuantumEntropyDetector(
            model=model,
            activation_name_func=self.resolve_name_func(),
            max_batch_size=self.train.max_batch_size,
            save_path=save_dir,
        )
