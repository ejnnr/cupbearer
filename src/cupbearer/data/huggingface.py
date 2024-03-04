from dataclasses import dataclass

import datasets
import torch

from . import DatasetConfig


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = datasets.load_dataset("imdb", split="train")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["text"], self.dataset[idx]["label"]


@dataclass
class IMDBDatasetConfig(DatasetConfig):
    @property
    def num_classes(self):
        return 2

    def _build(self):
        return IMDBDataset()
