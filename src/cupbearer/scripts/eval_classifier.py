import json

import lightning as L
from cupbearer.scripts.train_classifier import Classifier
from cupbearer.utils.scripts import run
from loguru import logger
from torch.utils.data import DataLoader

from .conf.eval_classifier_conf import Config


def main(cfg: Config):
    for trafo in cfg.data.get_transforms():
        logger.debug(f"Loading transform: {trafo}")
        trafo.load(cfg.dir.path)

    dataset = cfg.data.build()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.max_batch_size,
        shuffle=False,
    )

    assert cfg.dir.path is not None
    classifier = Classifier.load_from_checkpoint(
        cfg.dir.path / "last.ckpt", test_loader_names=["test"]
    )
    trainer = L.Trainer(
        logger=False,
        default_root_dir=cfg.dir.path,
    )
    metrics = trainer.test(classifier, [dataloader])

    with open(cfg.dir.path / "eval.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    run(main, Config)
