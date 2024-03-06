from dataclasses import dataclass, field

from torch.utils.data import Dataset

from cupbearer.utils import get_object

from .transforms import (
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
    ToTensor,
    Transform,
)


@dataclass(kw_only=True)
class PytorchDataset(Dataset):
    name: str
    train: bool = True
    transforms: list[Transform] = field(default_factory=lambda: [ToTensor()])
    default_augmentations: bool = True
    normalize: bool = False

    @property
    def raw_mean(self):
        raise NotImplementedError

    @property
    def raw_std(self):
        raise NotImplementedError

    def __post_init__(self):
        if self.normalize:
            self.transforms.append(Normalize(mean=self.raw_mean, std=self.raw_std))
        if self.default_augmentations and self.train:
            # Defaults from WaNet https://openreview.net/pdf?id=eEn8KTtJOx
            self.transforms.append(RandomCrop(p=0.8, padding=5))
            self.transforms.append(RandomRotation(p=0.5, degrees=10))

        self._dataset = self._build()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

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
class MNIST(PytorchDataset):
    name: str = "torchvision.datasets.MNIST"
    num_classes: int = 10
    normalize: bool = True

    @property
    def raw_mean(self):
        return (0.1307,)

    @property
    def raw_std(self):
        return (0.3081,)


@dataclass
class CIFAR10(PytorchDataset):
    name: str = "torchvision.datasets.CIFAR10"
    num_classes: int = 10
    normalize: bool = True

    @property
    def raw_mean(self):
        return (0.4914, 0.4822, 0.4465)

    @property
    def raw_std(self):
        return (0.247, 0.243, 0.261)

    def __post_init__(self):
        super().__post_init__()
        if self.default_augmentations and self.train:
            self.transforms.append(RandomHorizontalFlip(p=0.5))


@dataclass
class GTSRB(PytorchDataset):
    name: str = "torchvision.datasets.GTSRB"
    num_classes: int = 43
    transforms: list[Transform] = field(
        default_factory=lambda: [
            Resize(size=(32, 32)),
            ToTensor(),
        ]
    )
    normalize: bool = True

    @property
    def raw_mean(self):
        return (0.485, 0.456, 0.406)

    @property
    def raw_std(self):
        return (0.229, 0.224, 0.225)

    @property
    def _dataset_kws(self):
        # GTSRB takes split as keyword instead of train
        return dict(
            (key, val)
            if key != "train"
            else ("split", "train" if self.train else "test")
            for key, val in super()._dataset_kws.items()
        )
