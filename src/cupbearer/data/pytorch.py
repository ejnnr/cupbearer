from dataclasses import dataclass

from torch.utils.data import Dataset

from cupbearer.utils.utils import get_object, mutable_field

from . import DatasetConfig
from ._shared import ToNumpy, Transform


@dataclass(kw_only=True)
class PytorchConfig(DatasetConfig):
    name: str
    train: bool = True
    transforms: dict[str, Transform] = mutable_field({"to_numpy": ToNumpy()})

    def _build(self) -> Dataset:
        dataset_cls = get_object(self.name)
        # TODO: Do all torchvision datasets have these parameters?
        dataset = dataset_cls(  # type: ignore
            root="data", train=self.train, download=True
        )
        return dataset


@dataclass
class MNIST(PytorchConfig):
    name: str = "torchvision.datasets.MNIST"


@dataclass
class CIFAR10(PytorchConfig):
    name: str = "torchvision.datasets.CIFAR10"
