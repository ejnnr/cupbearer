"""Natural mechanism distinction tasks for 1L attention-only transformers.

This is just a light wrapper around datasets and models create by Jacob Hilton.
"""

import json
import os
from pathlib import Path
from typing import Any

import torch

from .task import Task


def decode_and_encode(tokens, model, new_model):
    decoded = model.tokenizer.decode(tokens)
    reencoded = new_model.tokenizer.encode(decoded)
    return reencoded

def pad_tokens(tokens, pad_token_id, max_len=16):
        tokens_len = len(tokens)
        return [pad_token_id] * (max(max_len - tokens_len, 0)) + tokens[-min(tokens_len, max_len):]

def get_effect_tokens(behavior_name, model):
    from hex_nn.masking.behaviors import registry as behavior_registry
    new_behavior = behavior_registry[behavior_name](model.tokenizer)
    new_effect_tokens = list(new_behavior.effect_tokens)
    return new_effect_tokens

def convert_task_to_model(behavior_name, new_model_name, task_data, model, new_model, cache_dir=None):
    # set up cache path
    cache_path = Path(cache_dir) / f"{behavior_name}_{new_model_name}_task.json" if cache_dir else None
    
    # try to load from cache
    if cache_path and cache_path.exists():
        with cache_path.open("r") as f:
            task_data = json.load(f)
            return task_data
    def decode_encode_data(data, model, new_model):
        return [
            {
                "prefix_tokens": decode_and_encode(example["prefix_tokens"], model, new_model),
                "completion_token": decode_and_encode([example["completion_token"]], model, new_model),
            }
            for example in data
        ]

    def pad_tokens_data(data, pad_token_id, max_len=16):
        return [
            {
                "prefix_tokens": pad_tokens(example.pop("prefix_tokens"), pad_token_id, max_len=max_len),
                **example
            }
            for example in data
        ]
    # decode and recode tokens
    task_data["train"] = decode_encode_data(task_data["train"], model, new_model)
    task_data["test_non_anomalous"] = decode_encode_data(task_data["test_non_anomalous"], model, new_model)
    task_data["test_anomalous"] = decode_encode_data(task_data["test_anomalous"], model, new_model)
    # pad tokens 
    task_data["train"] = pad_tokens_data(task_data["train"], model.tokenizer.pad_token_id)
    task_data["test_non_anomalous"] = pad_tokens_data(task_data["test_non_anomalous"], model.tokenizer.pad_token_id)
    task_data["test_anomalous"] = pad_tokens_data(task_data["test_anomalous"], model.tokenizer.pad_token_id)
    # add effect tokens
    new_effect_tokens = get_effect_tokens(behavior_name, model)
    task_data["effect_tokens"] = new_effect_tokens
    # save to cache
    if cache_path:
        with cache_path.open("w") as f:
            json.dump(task_data, f)
    return task_data  



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


def tiny_natural_mechanisms(name: str, device: str, new_model_name=None):
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

    cache_path = cache_dir / f"{name}_task.json"
    if cache_path.exists():
        with cache_path.open("r") as f:
            task_data = json.load(f)
    else:
        path = f"gs://arc-ml-public/distinctions/datasets/{name}_task.json"
        with bf.BlobFile(path) as f:
            task_data = json.load(f)
        with cache_path.open("w") as f:
            json.dump(task_data, f)
    
    if new_model_name is not None:
        new_model = HookedTransformer.from_pretrained(new_model_name).to(device)
        task_data = convert_task_to_model(name, new_model_name, task_data, model, new_model, cache_dir=cache_dir)
        model = new_model
    train_data = TinyNaturalMechanismsDataset(task_data["train"])
    normal_test_data = TinyNaturalMechanismsDataset(task_data["test_non_anomalous"])
    anomalous_test_data = TinyNaturalMechanismsDataset(task_data["test_anomalous"])

    return Task.from_separate_data(
        model=model,
        trusted_data=train_data,
        clean_test_data=normal_test_data,
        anomalous_test_data=anomalous_test_data,
    )
