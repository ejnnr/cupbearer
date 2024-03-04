import torch
from datasets import load_dataset

TAMPERING_DATSETS = {
    "diamonds": "redwoodresearch/diamonds-seed0",
    "text_props": "redwoodresearch/text_properties",
    "gen_stories": "redwoodresearch/generated_stories",
}


class TamperingDataset(torch.utils.data.Dataset):
    def __init__(self, name: str, train: bool = True):
        # TODO: allow for local loading / saving
        super().__init__()
        self.train = train
        self.name = name

        hf_name = (
            TAMPERING_DATSETS[self.name]
            if self.name in TAMPERING_DATSETS
            else self.name
        )
        split = "train" if self.train else "validation"
        self.dataset = load_dataset(hf_name, split=split)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return (
            sample["text"],
            torch.tensor([*sample["measurements"], all(sample["measurements"])]),
        )
        # sample["is_correct"], sample["is_clean"])

    def __len__(self):
        return len(self.dataset)
