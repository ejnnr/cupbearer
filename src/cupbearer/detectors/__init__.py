from cupbearer.utils.config_groups import register_config_group

from .abstraction import AbstractionDetectorConfig
from .config import DetectorConfig, StoredDetector
from .finetuning import FinetuningConfig
from .statistical import MahalanobisConfig, SpectralSignatureConfig, SpectreConfig

DETECTORS = {
    "abstraction": AbstractionDetectorConfig,
    "mahalanobis": MahalanobisConfig,
    "finetuning": FinetuningConfig,
    "from_run": StoredDetector,
    "spectral": SpectralSignatureConfig,
    "spectre": SpectreConfig,
}

register_config_group(DetectorConfig, DETECTORS)
