from dataclasses import dataclass

import hydra

from abstractions.utils.hydra import hydra_config
from . import DatasetConfig

from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


@hydra_config
@dataclass
class PytorchConfig(DatasetConfig):
    name: str
    train: bool = True

    def _get_dataset(self) -> Dataset:
        dataset_cls = hydra.utils.get_class(self.name)
        # TODO: Do all torchvision datasets have these parameters?
        dataset = dataset_cls(  # type: ignore
            root=to_absolute_path("data"), train=self.train, download=True
        )
        return dataset


@hydra_config
@dataclass
class MNIST(PytorchConfig):
    name: str = "torchvision.datasets.MNIST"


@hydra_config
@dataclass
class CIFAR10(PytorchConfig):
    name: str = "torchvision.datasets.CIFAR10"
