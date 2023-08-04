from cupbearer.utils.config_groups import register_config_group

from .abstraction import AbstractionDetectorConfig, AdversarialAbstractionConfig
from .config import DetectorConfig, StoredDetector
from .mahalanobis import MahalanobisConfig

DETECTORS = {
    "abstraction": AbstractionDetectorConfig,
    "adversarial_abstraction": AdversarialAbstractionConfig,
    "mahalanobis": MahalanobisConfig,
    "from_run": StoredDetector,
}

register_config_group(DetectorConfig, DETECTORS)
