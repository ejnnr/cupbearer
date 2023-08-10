from cupbearer.utils.config_groups import register_config_group, register_config_option

from ._shared import DatasetConfig, ToNumpy, TrainDataFromRun, Transform
from ._shared import TestDataConfig as TestDataConfig
from ._shared import TestDataMix as TestDataMix
from ._shared import numpy_collate as numpy_collate
from .adversarial import AdversarialExampleConfig
from .backdoors import CornerPixelBackdoor, NoiseBackdoor
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

# Need to import this after datasets and transforms have been registered,
# since BackdoorData uses them in config groups.
# noqa to prevent ruff from moving this line to the top.
from .backdoor_data import BackdoorData  # noqa

register_config_option(DatasetConfig, "backdoor", BackdoorData)
