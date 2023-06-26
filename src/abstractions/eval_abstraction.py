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
from abstractions.mahalanobis import MahalanobisDetector
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
    detector_run = Path(cfg.detector)
    # Note that hydra cwd management is disabled for this script, so no need for
    # to_absolute_path like in other files.
    path = detector_run / ".hydra" / "config.yaml"
    logger.info(f"Loading detector config from {path}")
    detector_cfg = OmegaConf.load(path)

    if cfg.debug:
        jax_config.update("jax_debug_nans", True)
        jax_config.update("jax_disable_jit", True)

    # Load the full model we want to abstract
    base_run = Path(detector_cfg.base_run)
    path = base_run / ".hydra" / "config.yaml"
    logger.info(f"Loading base model config from {path}")
    base_cfg = OmegaConf.load(path)

    full_computation = hydra.utils.call(base_cfg.model)
    full_model = abstraction.Model(computation=full_computation)
    full_params = utils.load(base_run / "model")["params"]

    # TODO: this should be more robust. Maybe AnomalyDetector should have a .load()
    # classmethod? (and checkpoints would need to store the type and some hparams)
    if "model" in detector_cfg:
        # We're dealing with an AbstractionDetector

        if detector_cfg.single_class:
            detector_cfg.model.output_dim = 2

        computation = hydra.utils.call(detector_cfg.model)
        # TODO: Might want to make this configurable somehow, but it's a reasonable
        # default for now
        maps = get_abstraction_maps(detector_cfg.model)
        model = abstraction.Abstraction(computation=computation, abstraction_maps=maps)

        detector = AbstractionDetector(
            model=full_model,
            params=full_params,
            abstraction=model,
            max_batch_size=cfg.max_batch_size,
            output_loss_fn=(
                single_class_loss_fn if detector_cfg.single_class else kl_loss_fn
            ),
        )
    elif "relative" in detector_cfg:
        # Mahalanobis detector
        detector = MahalanobisDetector(
            model=full_model,
            params=full_params,
            max_batch_size=cfg.max_batch_size,
        )
    else:
        raise ValueError("Unknown detector type")

    detector.load(detector_run / "detector")

    clean_dataset = data.get_pytorch_dataset(base_cfg.dataset, train=False)

    anomalous_datasets = {}

    anomalous_datasets = {
        k: data.get_dataset(dataset_cfg, base_run, base_cfg)
        for k, dataset_cfg in cfg.anomalies.items()
    }

    detector.eval(
        normal_dataset=clean_dataset,
        anomalous_datasets=anomalous_datasets,
        save_path=detector_run,
    )


if __name__ == "__main__":
    logger.remove()
    logger.level("METRICS", no=25, color="<green>", icon="📈")
    logger.add(
        sys.stdout, format="{level.icon} <level>{message}</level>", level="METRICS"
    )
    # Default logger for everything else:
    logger.add(sys.stdout, filter=lambda record: record["level"].name != "METRICS")
    evaluate()
