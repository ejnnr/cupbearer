import functools
import itertools
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from cupbearer import data
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
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
        # Need any seed so that labels are (somewhat) consistent over instances
        self._rng = np.random.default_rng(seed=5965)

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        if index >= self.length:
            raise IndexError
        return self.img, self._rng.integers(self.num_classes)


class TestMixedData:
    @staticmethod
    @pytest.fixture
    def clean_dataset():
        return DummyDataset(9, "a")

    @staticmethod
    @pytest.fixture
    def anomalous_dataset():
        return DummyDataset(7, "b")

    @staticmethod
    @pytest.fixture
    def mixed_dataset(clean_dataset, anomalous_dataset):
        return data.MixedData(clean_dataset, anomalous_dataset)

    @staticmethod
    def test_len(mixed_dataset):
        assert len(mixed_dataset) == 14
        assert mixed_dataset.normal_len == mixed_dataset.anomalous_len == 7

    @staticmethod
    def test_contents(mixed_dataset):
        for i in range(7):
            assert mixed_dataset[i] == ("a", 0)
        for i in range(7, 14):
            assert mixed_dataset[i] == ("b", 1)

    @staticmethod
    def test_uneven_weight(clean_dataset, anomalous_dataset):
        mixed_data = data.MixedData(clean_dataset, anomalous_dataset, normal_weight=0.3)
        # The 7 anomalous datapoints should be 70% of the dataset, so total
        # length should be 10.
        assert len(mixed_data) == 10
        assert mixed_data.normal_len == 3
        assert mixed_data.anomalous_len == 7
        for i in range(3):
            assert mixed_data[i] == ("a", 0)
        for i in range(3, 10):
            assert mixed_data[i] == ("b", 1)


class DatasetFixtures:
    @staticmethod
    @pytest.fixture
    def clean_image_dataset():
        return DummyImageData(9, 10, (8, 12))


class TestBackdoors(DatasetFixtures):
    @staticmethod
    @pytest.fixture(
        params=[
            data.backdoors.CornerPixelBackdoor,
            data.backdoors.NoiseBackdoor,
            functools.partial(data.backdoors.WanetBackdoor, path=None),
        ]
    )
    def backdoor_type(request):
        return request.param

    @staticmethod
    def test_backdoor_relabeling(clean_image_dataset, backdoor_type):
        target_class = 1
        dataset = data.BackdoorDataset(
            original=clean_image_dataset,
            backdoor=backdoor_type(p_backdoor=1.0, target_class=target_class),
        )
        for img, label in dataset:
            assert label == target_class

    @staticmethod
    def test_backdoor_img_changes(clean_image_dataset, backdoor_type):
        clean_data = data.BackdoorDataset(
            original=clean_image_dataset, backdoor=backdoor_type(p_backdoor=0.0)
        )
        anomalous_data = data.BackdoorDataset(
            original=clean_image_dataset, backdoor=backdoor_type(p_backdoor=1.0)
        )
        for clean_sample, (anomalous_img, _) in zip(clean_data, anomalous_data):
            clean_img, _ = clean_sample

            # Check that something has changed
            assert clean_img is not anomalous_data.backdoor(clean_sample)[0]
            assert torch.any(clean_img != anomalous_data.backdoor(clean_sample)[0])
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

    @staticmethod
    def test_wanet_backdoor(clean_image_dataset):
        # Pick a target class outside the actual range so we can later tell whether it
        # was set correctly.
        target_class = 10_000
        backdoor = data.backdoors.WanetBackdoor(
            path=None,
            p_backdoor=0.0,
            target_class=target_class,
        )
        clean_data = data.BackdoorDataset(
            original=clean_image_dataset,
            backdoor=backdoor,
        )
        anomalous_data = data.BackdoorDataset(
            original=clean_image_dataset,
            backdoor=backdoor.clone(
                p_backdoor=1.0,
            ),
        )
        noise_data = data.BackdoorDataset(
            original=clean_image_dataset,
            backdoor=backdoor.clone(
                p_backdoor=0.0,
                p_noise=1.0,
            ),
        )
        for (
            (clean_img, clean_label),
            (anoma_img, anoma_label),
            (noise_img, noise_label),
        ) in zip(clean_data, anomalous_data, noise_data):
            # Check labels. Our target class is outside the valid range,
            # so no chance it got randomly chosen.
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
        for ds1, ds2 in itertools.combinations(
            [clean_data, anomalous_data, noise_data],
            r=2,
        ):
            assert isinstance(ds1.backdoor, data.WanetBackdoor)
            assert isinstance(ds2.backdoor, data.WanetBackdoor)
            assert torch.allclose(
                ds1.backdoor.warping_field,
                ds2.backdoor.warping_field,
            )

    @staticmethod
    def test_wanet_backdoor_scale_invariance(clean_image_dataset):
        backdoor = data.backdoors.WanetBackdoor(
            path=None,
            p_backdoor=1.0,
        )
        mean = np.array([1, 2, 3])
        std = np.array([1 / 2, 1 / 4, 1 / 8])
        normalize = Normalize(mean=mean, std=std)
        denormalize = Normalize(
            mean=-mean * std**-1,
            std=std**-1,
        )
        for img, label in clean_image_dataset:
            torch.testing.assert_close(img, denormalize(normalize(img)))
            torch.testing.assert_close(
                backdoor((img, label))[0],
                denormalize(
                    backdoor(
                        (
                            normalize(img),
                            label,
                        )
                    )[0]
                ),
            )

    @staticmethod
    def test_wanet_backdoor_on_multiple_workers(
        clean_image_dataset,
    ):
        anomalous_data = data.BackdoorDataset(
            original=clean_image_dataset,
            backdoor=data.backdoors.WanetBackdoor(
                path=None, p_backdoor=1.0, p_noise=0.0
            ),
        )
        data_loader = DataLoader(dataset=anomalous_data, num_workers=2, batch_size=1)
        imgs = [img for img_batch, label_batch in data_loader for img in img_batch]
        assert all(torch.allclose(imgs[0], img) for img in imgs)

        clean_image = clean_image_dataset.img
        assert not any(torch.allclose(clean_image, img) for img in imgs)


@dataclass
class DummyPytorchDataset(data.PytorchDataset):
    name: str = "dummy"
    length: int = 32
    num_classes: int = 10
    shape: tuple[int, int] = (8, 12)
    default_augmentations: bool = True

    def __post_init__(self):
        # Because our data are already tensors, we need to disable the
        # default ToTensor
        assert len(self.transforms) == 1
        assert isinstance(self.transforms[0], data.transforms.ToTensor)
        self.transforms = []
        # Now call super to add the augmentations
        super().__post_init__()

    def _build(self) -> Dataset:
        return DummyImageData(self.length, self.num_classes, self.shape)


class TestAugmentations(DatasetFixtures):
    @staticmethod
    @pytest.fixture(
        params=[
            data.RandomCrop(padding=100),
            data.RandomHorizontalFlip(p=1.0),
            data.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
        ],
    )
    def augmentation(request):
        return request.param

    @staticmethod
    def test_augmentation(clean_image_dataset, augmentation):
        # See that augmentation does something unless dud
        for img, label in clean_image_dataset:
            aug_img, aug_label = augmentation((img, label))
            assert label == aug_label
            assert not torch.allclose(aug_img, img)

        # Try with multiple workers and batches
        data_loader = DataLoader(
            dataset=clean_image_dataset, num_workers=2, batch_size=3, drop_last=False
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

    @staticmethod
    def test_random_crop(clean_image_dataset):
        fill_val = 2.75
        augmentation = data.RandomCrop(
            padding=100,  # huge padding so that chance of no change is small
            fill=fill_val,
        )
        for img, label in clean_image_dataset:
            aug_img, aug_label = augmentation((img, label))
            assert torch.any(aug_img == fill_val)

    @staticmethod
    def test_pytorch_dataset_transforms():
        pytorch_dataset = DummyPytorchDataset()
        for (_img, _label), (img, label) in zip(
            pytorch_dataset._build(), pytorch_dataset
        ):
            assert _label == label
            assert _img.size() == img.size()
            assert _img is not img, "Transforms does not seem to have been applied"

        transforms = pytorch_dataset.transforms
        transform_typereps = [repr(type(t)) for t in transforms]
        augmentation_used = False
        for trafo in transforms:
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

    @staticmethod
    def test_no_augmentations():
        dataset = DummyPytorchDataset(default_augmentations=False)
        for trafo in dataset.transforms:
            assert not isinstance(trafo, data.transforms.ProbabilisticTransform)
