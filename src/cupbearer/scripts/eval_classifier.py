import json

import jax
from cupbearer.data import numpy_collate
from cupbearer.scripts.train_classifier import ClassificationTrainer
from cupbearer.utils.scripts import run
from loguru import logger
from torch.utils.data import DataLoader

from .conf.eval_classifier_conf import Config


def main(cfg: Config):
    # Need to load transforms *before* building the dataset.
    # Maybe? It should actually work either way.
    for trafo in cfg.data.get_transforms():
        logger.debug(f"Loading transform: {trafo}")
        trafo.load(cfg.dir.path)

    dataset = cfg.data.build()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.max_batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
    )

    # Dataloader returns images and labels, only images get passed to model
    images, _ = next(iter(dataloader))
    example_input = images[0:1]

    model = cfg.model.build_model()
    params = cfg.model.build_params()
    if params:
        override_variables = {"params": params}
    else:
        override_variables = None

    trainer = ClassificationTrainer(
        num_classes=cfg.num_classes,
        model=model,
        optimizer=None,
        example_input=example_input,
        log_dir=None,
        loggers=[],
        enable_progress_bar=cfg.pbar,
        rng=jax.random.PRNGKey(cfg.seed),
        override_variables=override_variables,
    )

    metrics = trainer.eval_model({"results": dataloader}, max_steps=cfg.max_steps)
    print(metrics)
    if cfg.dir.path:
        with open(cfg.dir.path / "eval.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    run(main, Config)
