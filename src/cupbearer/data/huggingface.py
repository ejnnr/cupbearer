import datasets
import torch


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool = True):
        split = "train" if train else "validation"
        self.dataset = datasets.load_dataset("imdb", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample["text"], sample["label"]
