# Make sure sources are added to register:
from ._shared import DatasetConfig, TrainDataFromRun, numpy_collate
from .adversarial import AdversarialExampleConfig
from .pytorch import PytorchConfig, MNIST, CIFAR10
