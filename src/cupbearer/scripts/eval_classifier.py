import json
from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, Dataset

from cupbearer.models import HookedModel
from cupbearer.scripts._shared import Classifier


def main(
    data: Dataset,
    model: HookedModel,
    path: Path | str,
    max_batches: Optional[int] = None,
    batch_size: int = 2048,
):
    path = Path(path)

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
    )

    classifier = Classifier.load_from_checkpoint(
        path / "checkpoints" / "last.ckpt",
        model=model,
        test_loader_names=["test"],
    )
    trainer = L.Trainer(
        logger=False,
        default_root_dir=path,
        limit_test_batches=max_batches,
    )
    metrics = trainer.test(classifier, [dataloader])

    with open(path / "eval.json", "w") as f:
        json.dump(metrics, f)
