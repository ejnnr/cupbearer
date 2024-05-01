"""Natural mechanism distinction tasks for 1L attention-only transformers.

This is just a light wrapper around datasets and models create by Jacob Hilton.
"""

import json
import os
from pathlib import Path
from typing import Any

import torch

from .task import Task


class TinyNaturalMechanismsDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the list to a tensor to make sure that pytorch's default collate_fn
        # batches the lists into a single tensor (TransformerLens requires that).
        # Note that all sequences have the same length in these datasets.
        return (
            torch.tensor(self.data[idx]["prefix_tokens"], dtype=torch.long),
            self.data[idx]["completion_token"],
        )


def tiny_natural_mechanisms_task(name: str, device: str):
    import blobfile as bf
    from transformer_lens import HookedTransformer

    # This seems to be necessary to access the public GCS files below without logging in
    os.environ["NO_GCE_CHECK"] = "true"

    model = HookedTransformer.from_pretrained(
        "attn-only-1l",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    ).to(device)

    # Downloading the models from GCS can take ~10 seconds, so we cache them locally.
    cache_dir = Path(".cupbearer_cache/tiny_natural_mechanisms/")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "main.pth"

    if cache_path.exists():
        state_dict = torch.load(cache_path, map_location=device)
    else:
        # `model_path` seems to have a typo, uses a `.path` extension instead of `.pth`
        # with bf.BlobFile(task_data["model_path"], "rb") as fh:
        with bf.BlobFile("gs://arc-ml-public/distinctions/models/main.pth", "rb") as fh:
            state_dict = torch.load(fh, map_location=device)
        state_dict["unembed.b_U"] = model.unembed.b_U
        torch.save(state_dict, cache_path)

    model.load_state_dict(state_dict)

    cache_path = Path(f".cupbearer_cache/arc/{name}_task.json")
    if cache_path.exists():
        with cache_path.open("r") as f:
            task_data = json.load(f)
    else:
        path = f"gs://arc-ml-public/distinctions/datasets/{name}_task.json"
        with bf.BlobFile(path) as f:
            task_data = json.load(f)
        with cache_path.open("w") as f:
            json.dump(task_data, f)

    train_data = TinyNaturalMechanismsDataset(task_data["train"])
    normal_test_data = TinyNaturalMechanismsDataset(task_data["test_non_anomalous"])
    anomalous_test_data = TinyNaturalMechanismsDataset(task_data["test_anomalous"])

    return Task.from_separate_data(
        model=model,
        trusted_data=train_data,
        clean_test_data=normal_test_data,
        anomalous_test_data=anomalous_test_data,
    )
