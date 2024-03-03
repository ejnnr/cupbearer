from dataclasses import dataclass
from typing import ClassVar

import torch
from datasets import load_dataset

from . import DatasetConfig


class TamperingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return (
            sample["text"],
            torch.tensor([*sample["measurements"], all(sample["measurements"])]),
        )
        # sample["is_correct"], sample["is_clean"])

    def __len__(self):
        return len(self.dataset)


TAMPERING_DATSETS = {
    "diamonds": "redwoodresearch/diamonds-seed0",
    "text_props": "redwoodresearch/text_properties",
    "gen_stories": "redwoodresearch/generated_stories",
}


@dataclass
class TamperingDataConfig(DatasetConfig):
    n_sensors: ClassVar[int] = 3  # not configurable
    train: bool = True  # TODO: how does cupbearer use this?
    name: str = None

    def __post_init__(self):
        assert self.name, "must pass name argument"
        return super().__post_init__()

    @property
    def num_classes(self):
        # only used for multi-class classification
        return None

    @property
    def num_labels(self):
        # n sensors + all(sensors)
        return self.n_sensors + 1

    def _build(self) -> TamperingDataset:  # TODO: allow for local loading / saving
        name = (
            TAMPERING_DATSETS[self.name]
            if self.name in TAMPERING_DATSETS
            else self.name
        )
        split = "train" if self.train else "validation"
        dataset = load_dataset(name, split=split)
        return TamperingDataset(dataset)
