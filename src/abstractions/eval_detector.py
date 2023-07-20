import copy
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

from abstractions import computations, data, utils
from abstractions.anomaly_detector import AnomalyDetector
from abstractions.computations import get_tau_maps, identity_init

CONFIG_NAME = Path(__file__).stem
utils.setup_hydra(CONFIG_NAME)


@hydra.main(
    version_base=None, config_path=f"conf/{CONFIG_NAME}", config_name=CONFIG_NAME
)
def main(cfg: DictConfig):
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

    # Load the full model we want to abstract
    base_run = Path(detector_cfg.base_run)
    path = base_run / ".hydra" / "config.yaml"
    logger.info(f"Loading base model config from {path}")
    base_cfg = OmegaConf.load(path)

    full_computation = hydra.utils.call(base_cfg.model)
    full_model = computations.Model(computation=full_computation)
    full_params = utils.load(base_run / "model")["params"]

    detector = AnomalyDetector.load(detector_run / "detector", full_model, full_params)

    # We want to use the test split of the training data as the reference distribution,
    # to make sure we at least don't flag that as anomalous.
    # TODO: this might not always work, dataset config might not have the a train/test
    # split. Also unclear whether this is the right way to do it, maybe we should just
    # have the test data as another anomalous dataloader if we care?
    reference_data = copy.deepcopy(base_cfg.train_data)
    reference_data.train = False
    # Remove backdoors
    reference_data.transforms = {}
    clean_dataset = data.get_dataset(reference_data, base_run)

    if cfg.adversarial:
        assert len(cfg.anomalies) == 1

        filter_maps = None
        if cfg.filter_maps:
            filter_maps = get_tau_maps(detector.model.computation, identity_init)

        new_dataset = data.get_dataset(
            next(iter(cfg.anomalies.values())),
            base_run,
            default_name=base_cfg.train_data.name,
        )

        with detector.adversarial(
            clean_dataset,
            new_dataset,
            filter_maps=filter_maps,
            new_batch_size=cfg.batch_size,
            normal_batch_size=cfg.normal_batch_size,
            num_epochs=cfg.num_epochs,
            normal_weight=cfg.normal_weight,
            clip=cfg.clip,
        ) as finetuned_vars:
            detector.eval(
                normal_dataset=clean_dataset,
                anomalous_datasets={"adversarial": new_dataset},
                save_path=detector_run,
            )
            utils.save(finetuned_vars, detector_run / "finetuned_vars", overwrite=True)

            return

    anomalous_datasets = {}

    anomalous_datasets = {
        k: data.get_dataset(
            dataset_cfg, base_run, default_name=base_cfg.train_data.name
        )
        for k, dataset_cfg in cfg.anomalies.items()
    }

    detector.eval(
        normal_dataset=clean_dataset,
        anomalous_datasets=anomalous_datasets,
        save_path=detector_run,
    )


if __name__ == "__main__":
    main()
