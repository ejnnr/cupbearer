from pathlib import Path

import torch
from torch.utils.data import Dataset

from cupbearer.data import make_adversarial_examples

from .task import Task


def adversarial_examples(
    model: torch.nn.Module,
    train_data: Dataset,
    test_data: Dataset,
    cache_path: Path,
    trusted_fraction: float = 1.0,
    clean_train_weight: float = 0.5,
    clean_test_weight: float = 0.5,
    **kwargs,
) -> Task:
    return Task.from_base_data(
        model=model,
        train_data=train_data,
        test_data=test_data,
        anomaly_func=lambda dataset, train: make_adversarial_examples(
            model,
            dataset,
            cache_path / f"adversarial_examples_{'train' if train else 'test'}",
            **kwargs,
        ),
        trusted_fraction=trusted_fraction,
        clean_train_weight=clean_train_weight,
        clean_test_weight=clean_test_weight,
    )
