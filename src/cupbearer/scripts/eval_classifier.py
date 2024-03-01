import json

import lightning as L
from loguru import logger
from torch.utils.data import DataLoader

from cupbearer.data import BackdoorDataset
from cupbearer.scripts._shared import Classifier
from cupbearer.utils.scripts import script

from .conf.eval_classifier_conf import Config


@script
def main(cfg: Config):
    assert cfg.path is not None  # make type checker happy

    if isinstance(cfg.data, BackdoorDataset):
        logger.debug(f"Loading transform: {cfg.data.backdoor}")
        cfg.data.backdoor.load(cfg.path)

    dataloader = DataLoader(
        cfg.data,
        batch_size=cfg.max_batch_size,
        shuffle=False,
    )

    classifier = Classifier.load_from_checkpoint(
        cfg.path / "checkpoints" / "last.ckpt",
        model=cfg.model,
        test_loader_names=["test"],
    )
    trainer = L.Trainer(
        logger=False,
        default_root_dir=cfg.path,
        limit_test_batches=cfg.max_batches,
    )
    metrics = trainer.test(classifier, [dataloader])

    with open(cfg.path / "eval.json", "w") as f:
        json.dump(metrics, f)
