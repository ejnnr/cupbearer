from dataclasses import dataclass

import numpy as np
import pytest
import torch
from cupbearer import data
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode


class DummyDataset(Dataset):
    def __init__(self, length: int, value: str):
        self.length = length
        self.value = value

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        return self.value


@dataclass
class DummyConfig(data.DatasetConfig):
    length: int
    value: str
    # Doesn't apply or matter
    num_classes: int = 0

    def _build(self) -> Dataset:
        return DummyDataset(self.length, self.value)


class DummyImageData(Dataset):
    def __init__(self, length: int, num_classes: int, shape: tuple[int, int]):
        self.length = length
        self.num_classes = num_classes
        self.img = torch.tensor(
            [
                [[i_y % 2, i_x % 2, (i_x + i_y + 1) % 2] for i_x in range(shape[1])]
                for i_y in range(shape[0])
            ],
            dtype=torch.float32,
            # Move channel dimension to front
        ).permute(2, 0, 1)
        # Need any seed so that labels are (somewhat) consitent over instances
        self._rng = np.random.default_rng(seed=5965)

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        if index >= self.length:
            raise IndexError
        return self.img, self._rng.integers(self.num_classes)


@dataclass
class DummyImageConfig(data.DatasetConfig):
    length: int
    num_classes: int = 10
    shape: tuple[int, int] = (8, 12)

    def _build(self) -> Dataset:
        return DummyImageData(self.length, self.num_classes, self.shape)


#########################
# Tests for TestDataMix
#########################


@pytest.fixture
def clean_dataset():
    return DummyDataset(9, "a")


@pytest.fixture
def anomalous_dataset():
    return DummyDataset(7, "b")


@pytest.fixture
def mixed_dataset(clean_dataset, anomalous_dataset):
    return data.MixedData(clean_dataset, anomalous_dataset)


@pytest.fixture
def clean_config():
    return DummyConfig(9, "a")


@pytest.fixture
def anomalous_config():
    return DummyConfig(7, "b")


@pytest.fixture
def mixed_config(clean_config, anomalous_config):
    return data.MixedDataConfig(clean_config, anomalous_config)


def test_len(mixed_dataset):
    assert len(mixed_dataset) == 14
    assert mixed_dataset.normal_len == mixed_dataset.anomalous_len == 7


def test_contents(mixed_dataset):
    for i in range(7):
        assert mixed_dataset[i] == ("a", 0)
    for i in range(7, 14):
        assert mixed_dataset[i] == ("b", 1)


def test_uneven_weight(clean_dataset, anomalous_dataset):
    mixed_data = data.MixedData(clean_dataset, anomalous_dataset, normal_weight=0.3)
    # The 7 anomalous datapoints should be 70% of the dataset, so total length should
    # be 10.
    assert len(mixed_data) == 10
    assert mixed_data.normal_len == 3
    assert mixed_data.anomalous_len == 7
    for i in range(3):
        assert mixed_data[i] == ("a", 0)
    for i in range(3, 10):
        assert mixed_data[i] == ("b", 1)


def test_simple_mixed_build(mixed_config):
    mixed_data = mixed_config.build()
    assert len(mixed_data) == 14
    assert mixed_data.normal_len == mixed_data.anomalous_len == 7
    for i in range(7):
        assert mixed_data[i] == ("a", 0)
    for i in range(7, 14):
        assert mixed_data[i] == ("b", 1)


def test_mixed_max_size(clean_config, anomalous_config):
    # Just some random big enough numbers:
    clean_config.length = 105
    anomalous_config.length = 97
    # These max sizes shouldn't affect anything, but why not throw them into the mix.
    clean_config.max_size = 51
    anomalous_config.max_size = 23
    # The actual mixed dataset we build now is the same as before: 10 datapoints,
    # 3 normal and 7 anomalous.
    mixed_config = data.MixedDataConfig(clean_config, anomalous_config)
    mixed_config.max_size = 10
    mixed_config.normal_weight = 0.3
    mixed_data = mixed_config.build()

    assert len(mixed_data) == 10
    assert mixed_data.normal_len == 3
    assert mixed_data.anomalous_len == 7
    for i in range(3):
        assert mixed_data[i] == ("a", 0)
    for i in range(3, 10):
        assert mixed_data[i] == ("b", 1)


#######################
# Tests for Backdoors
#######################


@pytest.fixture
def clean_image_config():
    return DummyImageConfig(9)


@pytest.fixture(
    params=[
        data.backdoors.CornerPixelBackdoor,
        data.backdoors.NoiseBackdoor,
        data.backdoors.WanetBackdoor,
    ]
)
def BackdoorConfig(request):
    return request.param


def test_backdoor_relabeling(clean_image_config, BackdoorConfig):
    clean_image_config.num_classes = 2**63 - 1
    target_class = 1
    data_config = data.BackdoorData(
        original=clean_image_config,
        backdoor=BackdoorConfig(
            p_backdoor=1.0,
            target_class=target_class,
        ),
    )
    for img, label in data_config.build():
        assert label == target_class


def test_backdoor_img_changes(clean_image_config, BackdoorConfig):
    clean_config = data.BackdoorData(
        original=clean_image_config,
        backdoor=BackdoorConfig(
            p_backdoor=0.0,
        ),
    )
    anomalous_config = data.BackdoorData(
        original=clean_image_config,
        backdoor=BackdoorConfig(
            p_backdoor=1.0,
        ),
    )
    for clean_sample, (anomalous_img, _) in zip(
        clean_config.build(),
        anomalous_config.build(),
    ):
        clean_img, _ = clean_sample

        # Check that something has changed
        assert clean_img is not anomalous_config.backdoor(clean_sample)[0]
        assert torch.any(clean_img != anomalous_config.backdoor(clean_sample)[0])
        assert torch.any(clean_img != anomalous_img)

        # Check that pixel values still in valid range
        assert torch.min(clean_img) >= 0
        assert torch.min(anomalous_img) >= 0
        assert torch.max(clean_img) <= 1
        assert torch.max(anomalous_img) <= 1

        # Check that backdoor overall applies a small change on average
        assert torch.mean(clean_img - anomalous_img) < (
            1.0 / np.sqrt(np.prod(clean_img.shape))
        )


def test_wanet_backdoor(clean_image_config):
    clean_image_config.num_classes = 2**63 - 1
    target_class = 1
    clean_config = data.BackdoorData(
        original=clean_image_config,
        backdoor=data.backdoors.WanetBackdoor(
            p_backdoor=0.0,
            target_class=target_class,
        ),
    )
    anomalous_config = data.BackdoorData(
        original=clean_image_config,
        backdoor=data.backdoors.WanetBackdoor(
            p_backdoor=1.0,
            target_class=target_class,
        ),
    )
    noise_config = data.BackdoorData(
        original=clean_image_config,
        backdoor=data.backdoors.WanetBackdoor(
            p_backdoor=0.0,
            p_noise=1.0,
            target_class=target_class,
        ),
    )
    for (
        (clean_img, clean_label),
        (anoma_img, anoma_label),
        (noise_img, noise_label),
    ) in zip(
        clean_config.build(),
        anomalous_config.build(),
        noise_config.build(),
    ):
        # Check labels
        assert clean_label != target_class
        assert anoma_label == target_class
        assert noise_label != target_class

        # Check that something has changed
        assert torch.any(clean_img != anoma_img)
        assert torch.any(clean_img != noise_img)
        assert torch.any(anoma_img != noise_img)

        # Check that pixel values still in valid range
        assert torch.min(clean_img) >= 0
        assert torch.min(anoma_img) >= 0
        assert torch.min(noise_img) >= 0
        assert torch.max(clean_img) <= 1
        assert torch.max(anoma_img) <= 1
        assert torch.max(noise_img) <= 1


def test_wanet_backdoor_on_multiple_workers(
    clean_image_config,
):
    clean_image_config.num_classes = 1
    target_class = 1
    anomalous_config = data.BackdoorData(
        original=clean_image_config,
        backdoor=data.backdoors.WanetBackdoor(
            p_backdoor=1.0,
            p_noise=0.0,
            target_class=target_class,
        ),
    )
    data_loader = DataLoader(
        dataset=anomalous_config.build(),
        num_workers=2,
        batch_size=1,
    )
    imgs = [img for img_batch, label_batch in data_loader for img in img_batch]
    assert all(torch.allclose(imgs[0], img) for img in imgs)

    clean_image = clean_image_config.build().dataset.img
    assert not any(torch.allclose(clean_image, img) for img in imgs)


#######################
# Tests for Augmentations
#######################


@pytest.fixture(
    params=[
        data.RandomCrop(padding=100),
        data.RandomHorizontalFlip(p=1.0),
        data.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
    ],
)
def augmentation(request):
    return request.param


def test_augmentation(clean_image_config, augmentation):
    # See that augmentation does something unless dud
    for img, label in clean_image_config.build():
        aug_img, aug_label = augmentation((img, label))
        assert label == aug_label
        assert not torch.allclose(aug_img, img)

    # Try with multiple workers and batches
    data_loader = DataLoader(
        dataset=clean_image_config.build(),
        num_workers=2,
        batch_size=3,
        drop_last=False,
    )
    for img, label in data_loader:
        aug_img, aug_label = augmentation((img, label))
        assert torch.all(label == aug_label)
        assert not torch.allclose(aug_img, img)

    # Test that updating p does change augmentation probability
    augmentation.p = 0.0
    for img, label in data_loader:
        aug_img, aug_label = augmentation((img, label))
        assert torch.all(label == aug_label)
        assert torch.all(aug_img == img)


def test_random_crop(clean_image_config):
    fill_val = 2.75
    augmentation = data.RandomCrop(
        padding=100,  # huge padding so that chance of no change is small
        fill=fill_val,
    )
    for img, label in clean_image_config.build():
        aug_img, aug_label = augmentation((img, label))
        assert torch.any(aug_img == fill_val)


@dataclass
class DummyPytorchImageConfig(data.PytorchConfig):
    name: str = "dummy"
    length: int = 32
    num_classes: int = 10
    shape: tuple[int, int] = (8, 12)

    def get_transforms(self):
        transforms = super().get_transforms()
        assert isinstance(transforms[0], data.transforms.ToTensor)
        return transforms[1:]

    def _build(self) -> Dataset:
        return DummyImageData(self.length, self.num_classes, self.shape)


@pytest.fixture
def pytorch_data_config():
    return DummyPytorchImageConfig()


def test_pytorch_dataset_transforms(pytorch_data_config, BackdoorConfig):
    for (_img, _label), (img, label) in zip(
        pytorch_data_config._build(), pytorch_data_config.build()
    ):
        assert _label == label
        assert _img.size() == img.size()
        assert _img is not img, "Transforms does not seem to have been applied"

    transforms = pytorch_data_config.get_transforms()
    transform_typereps = [repr(type(t)) for t in transforms]
    augmentation_used = False
    for trafo in pytorch_data_config.get_transforms():
        # Check that transform is unique in list
        assert transforms.count(trafo) == 1
        assert transform_typereps.count(repr(type(trafo))) == 1

        # Check transform types
        assert isinstance(trafo, data.transforms.Transform)
        if isinstance(trafo, data.transforms.ProbabilisticTransform):
            augmentation_used = True
        else:
            # Augmentations should by default come after all base transforms
            assert not augmentation_used, "Transform applied after augmentation"
    assert augmentation_used

    # Test for BackdoorData
    data_config = data.BackdoorData(
        original=pytorch_data_config,
        backdoor=BackdoorConfig(),
    )
    transforms = data_config.get_transforms()
    transform_typereps = [repr(type(t)) for t in transforms]
    augmentation_used = False
    backdoor_used = False
    for trafo in data_config.get_transforms():
        # Check that transform is unique in list
        assert transforms.count(trafo) == 1
        assert transform_typereps.count(repr(type(trafo))) == 1

        # Check transform types
        assert not backdoor_used, "Multiple backdoors in transforms"
        assert isinstance(trafo, data.transforms.Transform)
        if isinstance(trafo, data.transforms.ProbabilisticTransform):
            augmentation_used = True
        elif isinstance(trafo, data.backdoors.Backdoor):
            backdoor_used = True
        else:
            assert not augmentation_used, "Transform applied after augmentation"
    assert augmentation_used
    assert backdoor_used


def test_no_augmentations(BackdoorConfig):
    pytorch_data_config = DummyPytorchImageConfig(default_augmentations=False)
    data_config = data.BackdoorData(
        original=pytorch_data_config,
        backdoor=BackdoorConfig(),
    )
    for trafo in data_config.get_transforms():
        assert not isinstance(trafo, data.transforms.ProbabilisticTransform)
