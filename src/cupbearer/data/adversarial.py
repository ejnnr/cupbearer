import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from torch.utils.data import Dataset

from cupbearer.utils import utils

from . import DatasetConfig, TrainDataFromRun


@dataclass
class AdversarialExampleConfig(DatasetConfig, utils.PathConfigMixin):
    attack_batch_size: Optional[int] = None
    success_threshold: float = 0.1
    steps: int = 40

    def _build(self) -> Dataset:
        return AdversarialExampleDataset(
            base_run=self.get_path(),
            num_examples=self.max_size,
            attack_batch_size=self.attack_batch_size,
            success_threshold=self.success_threshold,
            steps=self.steps,
        )

    @property
    def num_classes(self):
        data_cfg = TrainDataFromRun(path=self.get_path())
        return data_cfg.num_classes

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.attack_batch_size = 2
            self.success_threshold = 1.0


class AdversarialExampleDataset(Dataset):
    def __init__(
        self,
        base_run,
        num_examples=None,
        attack_batch_size: Optional[int] = None,
        success_threshold: float = 0.1,
        steps: int = 40,
    ):
        base_run = Path(base_run)
        self.base_run = base_run
        if not (base_run / "adv_examples").exists():
            logger.info(
                "Adversarial examples not found, running attack with default settings"
            )
            command = [
                "python",
                "-m",
                "cupbearer.scripts.make_adversarial_examples",
                "--dir.full",
                str(base_run),
                "--success_threshold",
                str(success_threshold),
                "--steps",
                str(steps),
            ]
            if num_examples is not None:
                command += ["--max_examples", str(num_examples)]
            if attack_batch_size is not None:
                command += ["--batch_size", str(attack_batch_size)]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(str(e.stdout).replace("\\n", "\n"))
                print(str(e.stderr).replace("\\n", "\n"))
                raise e
            else:
                print(result.stdout)
                print(result.stderr)

        data = utils.load(base_run / "adv_examples")
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
