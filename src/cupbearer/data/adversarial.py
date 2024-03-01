import os
from pathlib import Path
from typing import Optional

import torch
import torchattacks
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset

from cupbearer.utils import utils


class AdversarialExampleDataset(Dataset):
    def __init__(self, advexes: torch.Tensor, labels: torch.Tensor):
        self.advexes = advexes
        self.labels = labels

    @classmethod
    def from_file(cls, filepath: Path, num_examples=None):
        data = utils.load(filepath)
        assert isinstance(data, dict)
        advexes = data["adv_inputs"]
        labels = data["labels"]

        if num_examples is None:
            num_examples = len(advexes)
        if len(advexes) < num_examples:
            raise ValueError(
                f"Only {len(advexes)} adversarial examples exist, "
                f"but {num_examples} were requested"
            )

        return cls(advexes[:num_examples], labels[:num_examples])

    def __len__(self):
        return len(self.advexes)

    def __getitem__(self, idx):
        # Labels are the original ones. We need to return them mainly for implementation
        # reasons: for eval, normal and anomalous data will be batched together, so
        # since the normal data includes labels, the anomalous one needs to as well.
        # TODO: Probably detectors should just never have access to labels during evals
        # (none of the current ones make use of them anyway). If a detector needs them,
        # it should use the model-generated labels, not ground truth ones.
        return self.advexes[idx], int(self.labels[idx])


def make_adversarial_examples(
    model: torch.nn.Module,
    dataset: Dataset,
    save_path: Path | str,
    batch_size: int = 128,
    eps: float = 8 / 255,
    max_examples: Optional[int] = None,
    success_threshold: float = 0.1,
    steps: int = 40,
) -> AdversarialExampleDataset:
    save_path = Path(save_path).with_suffix(".pt")
    if os.path.exists(save_path):
        logger.info("Adversarial examples already exist, skipping attack")
        return AdversarialExampleDataset.from_file(save_path, num_examples=max_examples)

    if max_examples:
        dataset = Subset(dataset, range(max_examples))

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
        # Make sure we delete the unsuccessful data so we don't load it later
        save_path.unlink()
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
    plt.savefig(save_path.with_suffix(".pdf"))

    return AdversarialExampleDataset.from_file(save_path)
