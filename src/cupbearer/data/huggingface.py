import datasets
import torch


class HuggingfaceDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, text_key="text", label_key="label"):
        self.hf_dataset = hf_dataset
        self.text_key = text_key
        self.label_key = label_key

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        return sample[self.text_key], sample[self.label_key]

    def __repr__(self):
        return (
            f"HuggingfaceDataset(hf_dataset={self.hf_dataset}, "
            f"text_key={self.text_key}, label_key={self.label_key})"
        )


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool = True):
        split = "train" if train else "validation"
        self.dataset = datasets.load_dataset("imdb", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample["text"], sample["label"]
