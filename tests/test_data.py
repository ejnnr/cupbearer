from dataclasses import dataclass

import numpy as np
import pytest

# We shouldn't import TestDataMix directly because that will make pytest think
# it's a test.
from cupbearer import data
from torch.utils.data import Dataset


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
        self.img = np.array(
            [
                [[i_y % 2, i_x % 2, (i_x + i_y) % 2] for i_x in range(shape[1])]
                for i_y in range(shape[0])
            ],
            dtype=np.float32,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> tuple[np.ndarray, int]:
        if index >= self.length:
            raise IndexError
        return self.img, np.random.randint(self.num_classes)


@dataclass
class DummyImageConfig(data.DatasetConfig):
    length: int
    num_classes: int = 10
    shape: tuple[int, int] = (8, 8)

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
    return data.TestDataMix(clean_dataset, anomalous_dataset)


@pytest.fixture
def clean_config():
    return DummyConfig(9, "a")


@pytest.fixture
def anomalous_config():
    return DummyConfig(7, "b")


@pytest.fixture
def mixed_config(clean_config, anomalous_config):
    return data.TestDataConfig(clean_config, anomalous_config)


def test_len(mixed_dataset):
    assert len(mixed_dataset) == 14
    assert mixed_dataset.normal_len == mixed_dataset.anomalous_len == 7


def test_contents(mixed_dataset):
    for i in range(7):
        assert mixed_dataset[i] == ("a", 0)
    for i in range(7, 14):
        assert mixed_dataset[i] == ("b", 1)


def test_uneven_weight(clean_dataset, anomalous_dataset):
    mixed_data = data.TestDataMix(clean_dataset, anomalous_dataset, normal_weight=0.3)
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
    mixed_config = data.TestDataConfig(clean_config, anomalous_config)
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
    for (clean_img, _), (anomalous_img, _) in zip(
        clean_config.build(),
        anomalous_config.build(),
    ):
        # Check that something has changed
        assert not np.all(clean_img == anomalous_img)

        # Check that pixel values still in valid range
        assert np.min(clean_img) >= 0
        assert np.min(anomalous_img) >= 0
        assert np.max(clean_img) <= 1
        assert np.max(anomalous_img) <= 1

        # Check that backdoor overall applies a small change on average
        assert np.mean(clean_img - anomalous_img) < (
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
        assert np.any(clean_img != anoma_img)
        assert np.any(clean_img != noise_img)
        assert np.any(anoma_img != noise_img)

        # Check that pixel values still in valid range
        assert np.min(clean_img) >= 0
        assert np.min(anoma_img) >= 0
        assert np.min(noise_img) >= 0
        assert np.max(clean_img) <= 1
        assert np.max(anoma_img) <= 1
        assert np.max(noise_img) <= 1
