import json
from pathlib import Path
from typing import Optional

import lightning as L
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from cupbearer.data import BackdoorDataset
from cupbearer.models import HookedModel
from cupbearer.scripts._shared import Classifier
from cupbearer.utils.scripts import script


@script
def main(
    data: Dataset,
    model: HookedModel,
    path: Path | str,
    max_batches: Optional[int] = None,
    max_batch_size: int = 2048,
):
    path = Path(path)

    if isinstance(data, BackdoorDataset):
        logger.debug(f"Loading transform: {data.backdoor}")
        data.backdoor.load(path)

    dataloader = DataLoader(
        data,
        batch_size=max_batch_size,
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
