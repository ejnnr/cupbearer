from cupbearer.utils.config_groups import register_config_group, register_config_option

from ._shared import DatasetConfig, ToTensor, TrainDataFromRun, Transform
from ._shared import TestDataConfig as TestDataConfig
from ._shared import TestDataMix as TestDataMix
from .adversarial import AdversarialExampleConfig
from .backdoors import CornerPixelBackdoor, NoiseBackdoor, WanetBackdoor
from .pytorch import CIFAR10, GTSRB, MNIST, PytorchConfig
from .toy_ambiguous_features import ToyFeaturesConfig

DATASETS = {
    "pytorch": PytorchConfig,
    "mnist": MNIST,
    "cifar10": CIFAR10,
    "gtsrb": GTSRB,
    "from_run": TrainDataFromRun,
    "adversarial": AdversarialExampleConfig,
    "toy_features": ToyFeaturesConfig,
}

TRANSFORMS = {
    "to_tensor": ToTensor,
    "corner": CornerPixelBackdoor,
    "noise": NoiseBackdoor,
    "wanet": WanetBackdoor,
}

register_config_group(DatasetConfig, DATASETS)
register_config_group(Transform, TRANSFORMS)

# Need to import this after datasets and transforms have been registered,
# since BackdoorData uses them in config groups.
# noqa to prevent ruff from moving this line to the top.
from .backdoor_data import BackdoorData  # noqa

register_config_option(DatasetConfig, "backdoor", BackdoorData)
