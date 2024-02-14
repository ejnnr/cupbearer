from ._shared import DatasetConfig as DatasetConfig
from ._shared import TestDataConfig as TestDataConfig
from ._shared import TestDataMix as TestDataMix
from ._shared import TrainDataFromRun
from .adversarial import AdversarialExampleConfig
from .backdoor_data import BackdoorData as BackdoorData
from .backdoors import Backdoor as Backdoor
from .backdoors import CornerPixelBackdoor, NoiseBackdoor, WanetBackdoor
from .pytorch import CIFAR10, GTSRB, MNIST, PytorchConfig
from .toy_ambiguous_features import ToyFeaturesConfig
from .transforms import (
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
    ToTensor,
    Transform,
)

DATASETS = {
    "pytorch": PytorchConfig,
    "mnist": MNIST,
    "cifar10": CIFAR10,
    "gtsrb": GTSRB,
    "from_run": TrainDataFromRun,
    "adversarial": AdversarialExampleConfig,
    "toy_features": ToyFeaturesConfig,
}

TRANSFORMS: dict[str, type[Transform]] = {
    "to_tensor": ToTensor,
    "resize": Resize,
    "random_crop": RandomCrop,
    "random_rotation": RandomRotation,
    "random_horizontal_flip": RandomHorizontalFlip,
}

BACKDOORS = {
    "corner": CornerPixelBackdoor,
    "noise": NoiseBackdoor,
    "wanet": WanetBackdoor,
}
