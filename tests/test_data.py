from dataclasses import dataclass

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
