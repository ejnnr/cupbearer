from abc import ABC
from dataclasses import dataclass, field

from cupbearer.detectors.config import ActivationBasedDetectorConfig
from cupbearer.utils.utils import BaseConfig

from .mahalanobis_detector import MahalanobisDetector
from .spectral_detector import SpectralSignatureDetector
from .spectre_detector import SpectreDetector


@dataclass
class ActivationConvolutionTrainConfig(BaseConfig, ABC):
    max_batches: int = 0
    batch_size: int = 4096
    max_batch_size: int = 4096
    pbar: bool = True
    debug: bool = False
    rcond: float = 1e-5
    # robust: bool = False  # TODO spectre uses
    # https://www.semanticscholar.org/paper/Being-Robust-(in-High-Dimensions)-Can-Be-Practical-Diakonikolas-Kamath/2a6de51d86f13e9eb7efa85491682dad0ccd65e8?utm_source=direct_link

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.max_batches = 3
            self.batch_size = 5


@dataclass
class MahalanobisTrainConfig(ActivationConvolutionTrainConfig):
    relative: bool = False


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
    train: ActivationConvolutionTrainConfig = field(
        default_factory=ActivationConvolutionTrainConfig
    )

    def build(self, model, save_dir) -> SpectralSignatureDetector:
        return SpectralSignatureDetector(
            model=model,
            activation_name_func=self.resolve_name_func(),
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )


@dataclass
class SpectreConfig(ActivationBasedDetectorConfig):
    train: ActivationConvolutionTrainConfig = field(
        default_factory=ActivationConvolutionTrainConfig
    )

    def build(self, model, save_dir) -> SpectralSignatureDetector:
        return SpectreDetector(
            model=model,
            activation_name_func=self.resolve_name_func(),
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )
