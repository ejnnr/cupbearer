import json

import lightning as L
from cupbearer.scripts._shared import Classifier
from cupbearer.utils.scripts import run
from loguru import logger
from torch.utils.data import DataLoader

from .conf.eval_classifier_conf import Config


def main(cfg: Config):
    assert cfg.data is not None  # make type checker happy
    assert cfg.path is not None  # make type checker happy

    for trafo in cfg.data.get_transforms():
        logger.debug(f"Loading transform: {trafo}")
        trafo.load(cfg.path)

    dataset = cfg.data.build()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.max_batch_size,
        shuffle=False,
    )

    classifier = Classifier.load_from_checkpoint(
        cfg.path / "checkpoints" / "last.ckpt", test_loader_names=["test"]
    )
    trainer = L.Trainer(
        logger=False,
        default_root_dir=cfg.path,
        limit_test_batches=cfg.max_batches,
    )
    metrics = trainer.test(classifier, [dataloader])

    with open(cfg.path / "eval.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    run(main, Config)
