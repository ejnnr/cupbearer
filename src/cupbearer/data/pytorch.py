from dataclasses import dataclass

from torch.utils.data import Dataset

from cupbearer.utils.utils import get_object, mutable_field

from . import DatasetConfig
from ._shared import ToNumpy, Transform, Resize


@dataclass(kw_only=True)
class PytorchConfig(DatasetConfig):
    name: str
    train: bool = True
    transforms: dict[str, Transform] = mutable_field({"to_numpy": ToNumpy()})

    @property
    def _dataset_kws(self):
        '''The keyword arguments passed to the dataset constructor.'''
        return {
            "root": "data",
            "train": self.train,
            "download": True,
        }

    def _build(self) -> Dataset:
        dataset_cls = get_object(self.name)
        dataset = dataset_cls(**self._dataset_kws)  # type: ignore
        return dataset


@dataclass
class MNIST(PytorchConfig):
    name: str = "torchvision.datasets.MNIST"


@dataclass
class CIFAR10(PytorchConfig):
    name: str = "torchvision.datasets.CIFAR10"


@dataclass
class GTSRB(PytorchConfig):
    name: str = "torchvision.datasets.GTSRB"
    transforms: dict[str, Transform] = mutable_field({
        "resize": Resize(size=(32, 32)),
        "to_numpy": ToNumpy(),
    })

    @property
    def _dataset_kws(self):
        # GTSRB takes split as keyword instead of train
        return dict(
            (key, val) if key != "train"
            else ("split", "train" if self.train else "test")
            for key, val in super()._dataset_kws.items()
        )
