import json
from pathlib import Path
import sys
import hydra
from jax.config import config as jax_config
from loguru import logger
from matplotlib import pyplot as plt
import sklearn.metrics

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from iceberg import Bounds, Renderer, Colors
from iceberg.primitives import Blank

from abstractions import abstraction, data, utils
from abstractions.adversarial_examples import AdversarialExampleDataset
from abstractions.computations import get_abstraction_maps
from abstractions.train_abstraction import (
    AbstractionDetector,
    AbstractionTrainer,
    compute_losses,
    kl_loss_fn,
    single_class_loss_fn,
)


@hydra.main(version_base=None, config_path="conf", config_name="eval_abstraction")
def evaluate(cfg: DictConfig):
    """Execute model evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    train_run = Path(cfg.train_run)
    # Note that hydra cwd management is disabled for this script, so no need for
    # to_absolute_path like in other files.
    path = train_run / ".hydra" / "config.yaml"
    logger.info(f"Loading abstraction config from {path}")
    train_cfg = OmegaConf.load(path)

    if cfg.debug:
        jax_config.update("jax_debug_nans", True)
        jax_config.update("jax_disable_jit", True)

    # Load the full model we want to abstract
    base_run = Path(train_cfg.base_run)
    path = base_run / ".hydra" / "config.yaml"
    logger.info(f"Loading base model config from {path}")
    base_cfg = OmegaConf.load(path)

    full_computation = hydra.utils.call(base_cfg.model)
    full_model = abstraction.Model(computation=full_computation)
    full_params = utils.load(base_run / "model.pytree")["params"]

    clean_dataset = data.get_dataset(base_cfg.dataset, train=False)
    # First sample, only input without label and info. Also need to add a batch dimension
    example_input = clean_dataset[0][0][None]
    _, example_activations = full_model.apply(
        {"params": full_params}, example_input, return_activations=True
    )

    if train_cfg.single_class:
        train_cfg.model.output_dim = 2

    computation = hydra.utils.call(train_cfg.model)
    # TODO: Might want to make this configurable somehow, but it's a reasonable
    # default for now
    maps = get_abstraction_maps(train_cfg.model)
    model = abstraction.Abstraction(computation=computation, abstraction_maps=maps)

    trainer = AbstractionTrainer(
        model=model,
        output_loss_fn=single_class_loss_fn if train_cfg.single_class else kl_loss_fn,
        optimizer=hydra.utils.instantiate(train_cfg.optim),
        example_input=example_activations,
        log_dir=train_run,
        check_val_every_n_epoch=1,
        loggers=[],
        enable_progress_bar=False,
    )
    trainer.load_model()

    detector = AbstractionDetector(
        model=full_model,
        params=full_params,
        trainer=trainer,
        max_batch_size=cfg.max_batch_size,
    )
    # TODO: this is very hacky, there should be a standardized way to load detectors from disk
    detector.trained = True

    match cfg.anomaly:
        case "backdoor":
            anomalous_dataset = data.get_dataset(
                base_cfg.dataset,
                train=False,
                transforms=data.get_transforms({"pixel_backdoor": {"p_backdoor": 1.0}}),
            )

        case "different_corner":
            anomalous_dataset = data.get_dataset(
                base_cfg.dataset,
                train=False,
                transforms=data.get_transforms(
                    {"pixel_backdoor": {"p_backdoor": 1.0, "corner": "top-left"}}
                ),
            )

        case "gaussian_noise":
            anomalous_dataset = data.get_dataset(
                base_cfg.dataset,
                train=False,
                transforms=data.get_transforms({"noise": {"std": 0.3}}),
            )

        case "adversarial":
            anomalous_dataset = AdversarialExampleDataset(base_run)

        case _:
            raise ValueError(f"Unknown anomaly type {cfg.anomaly}")

    detector.eval(
        normal_dataset=clean_dataset,
        anomalous_dataset=anomalous_dataset,
        save_path=train_run,
    )

    trainer.close_loggers()


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, color="<green>", icon="📈")
    logger.add(
        sys.stdout, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stdout, filter=lambda record: record["level"].name != "METRICS")
    evaluate()
