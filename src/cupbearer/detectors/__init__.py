from .abstraction import AbstractionDetectorConfig
from .config import DetectorConfig as DetectorConfig
from .config import StoredDetector
from .finetuning import FinetuningConfig
from .mahalanobis import MahalanobisConfig

DETECTORS = {
    "abstraction": AbstractionDetectorConfig,
    "mahalanobis": MahalanobisConfig,
    "finetuning": FinetuningConfig,
    "from_run": StoredDetector,
}
