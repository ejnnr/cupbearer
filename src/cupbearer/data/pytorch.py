from dataclasses import dataclass

from torch.utils.data import Dataset

from cupbearer.utils.utils import get_object, mutable_field

from . import DatasetConfig
from .transforms import Resize, ToTensor, Transform


@dataclass(kw_only=True)
class PytorchConfig(DatasetConfig):
    name: str
    # This is an abstractproperty on the parent class, but it's a bit more
    # convenient to just make it a field here.
    num_classes: int
    train: bool = True
    transforms: dict[str, Transform] = mutable_field({"to_tensor": ToTensor()})
    augmentations: dict[str, Transform] = mutable_field(
        {
            # "random_crop":  # TODO
            # "random_rotation":  # TODO
        }
    )

    @property
    def _dataset_kws(self):
        """The keyword arguments passed to the dataset constructor."""
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
    num_classes: int = 10


@dataclass
class CIFAR10(PytorchConfig):
    name: str = "torchvision.datasets.CIFAR10"
    num_classes: int = 10
    augmentations: dict[str, Transform] = mutable_field(
        {
            # "random_crop":  # TODO
            # "random_rotation":  # TODO
            # **{"random_horizontal_flip": None},  # TODO
        }
    )


@dataclass
class GTSRB(PytorchConfig):
    name: str = "torchvision.datasets.GTSRB"
    num_classes: int = 43
    transforms: dict[str, Transform] = mutable_field(
        {
            "resize": Resize(size=(32, 32)),
            "to_tensor": ToTensor(),
        }
    )

    @property
    def _dataset_kws(self):
        # GTSRB takes split as keyword instead of train
        return dict(
            (key, val)
            if key != "train"
            else ("split", "train" if self.train else "test")
            for key, val in super()._dataset_kws.items()
        )
