import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchattacks
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset

from cupbearer.models import StoredModel
from cupbearer.utils import utils

from . import DatasetConfig, TrainDataFromRun


def make_adversarial_example(
    path: Path,
    filename: str,
    batch_size: int = 128,
    eps: float = 8 / 255,
    max_examples: Optional[int] = None,
    success_threshold: float = 0.1,
    steps: int = 40,
    use_test_data: bool = False,
):
    save_path = path / f"{filename}.pt"
    if os.path.exists(save_path):
        logger.info("Adversarial examples already exist, skipping attack")
        return
    else:
        logger.info(
            "Adversarial examples not found, running attack with default settings"
        )

    model_cfg = StoredModel(path=path)
    data_cfg = TrainDataFromRun(path=path)
    if use_test_data:
        data_cfg = data_cfg.get_test_split()

    dataset = data_cfg.build()
    if max_examples:
        dataset = Subset(dataset, range(max_examples))
    image, _ = dataset[0]
    model = model_cfg.build_model(input_shape=image.shape)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    atk = torchattacks.PGD(
        model, eps=eps, alpha=2 / 255, steps=steps, random_start=True
    )
    rob_acc, l2, elapsed_time = atk.save(dataloader, save_path, return_verbose=True)

    # N.B. rob_acc is in percent while success_threshold is not
    if rob_acc > 100 * success_threshold:
        raise RuntimeError(
            "Attack failed, new accuracy is"
            f" {rob_acc}% > {100 * success_threshold}%."
        )

    # Plot a few adversarial examples in a grid and save the plot as a pdf
    adv_examples = torch.load(save_path)["adv_inputs"]
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        ax = axs[i // 3, i % 3]
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            ax.imshow(adv_examples[i].permute(1, 2, 0))
        except IndexError:
            pass
    plt.tight_layout()
    plt.savefig(path / "adv_examples.pdf")


@dataclass
class AdversarialExampleConfig(DatasetConfig):
    path: Path
    attack_batch_size: int = 128
    success_threshold: float = 0.1
    steps: int = 40
    eps: float = 8 / 255
    use_test_data: bool = False

    def _build(self) -> Dataset:
        filename = f"adv_examples_{'test' if self.use_test_data else 'train'}"
        make_adversarial_example(
            path=self.path,
            filename=filename,
            batch_size=self.attack_batch_size,
            eps=self.eps,
            max_examples=self.max_size,
            success_threshold=self.success_threshold,
            steps=self.steps,
            use_test_data=self.use_test_data,
        )

        return AdversarialExampleDataset(
            filepath=self.path / filename, num_examples=self.max_size
        )

    @property
    def num_classes(self):
        data_cfg = TrainDataFromRun(path=self.path)
        return data_cfg.num_classes


class AdversarialExampleDataset(Dataset):
    def __init__(self, filepath: Path, num_examples=None):
        data = utils.load(filepath)
        assert isinstance(data, dict)
        self.examples = data["adv_inputs"]
        self.labels = data["labels"]

        if num_examples is None:
            num_examples = len(self.examples)
        self.num_examples = num_examples
        if len(self.examples) < num_examples:
            raise ValueError(
                f"Only {len(self.examples)} adversarial examples exist, "
                f"but {num_examples} were requested"
            )

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        if idx >= self.num_examples:
            raise IndexError(f"Index {idx} is out of range")
        # Labels are the original ones. We need to return them mainly for implementation
        # reasons: for eval, normal and anomalous data will be batched together, so
        # since the normal data includes labels, the anomalous one needs to as well.
        # TODO: Probably detectors should just never have access to labels during evals
        # (none of the current ones make use of them anyway). If a detector needs them,
        # it should use the model-generated labels, not ground truth ones.
        return self.examples[idx], int(self.labels[idx])
