# Make sure sources are added to register:
from abstractions.utils.config_groups import register_config_group

from ._shared import DatasetConfig, ToNumpy, TrainDataFromRun, Transform
from ._shared import numpy_collate as numpy_collate
from .adversarial import AdversarialExampleConfig
from .backdoor import CornerPixelBackdoor, NoiseBackdoor
from .pytorch import CIFAR10, MNIST, PytorchConfig

DATASETS = {
    "pytorch": PytorchConfig,
    "mnist": MNIST,
    "cifar10": CIFAR10,
    "from_run": TrainDataFromRun,
    "adversarial": AdversarialExampleConfig,
}

TRANSFORMS = {
    "to_numpy": ToNumpy,
    "corner": CornerPixelBackdoor,
    "noise": NoiseBackdoor,
}

register_config_group(DatasetConfig, DATASETS)
register_config_group(Transform, TRANSFORMS)
