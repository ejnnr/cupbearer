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
    save_path = None
    if cfg.detector_path is not None:
        detector_run = Path(cfg.detector_path)
        save_path = Path(cfg.detector_path)

        # Note that hydra cwd management is disabled for this script, so no need for
        # to_absolute_path like in other files.
        path = detector_run / ".hydra" / "config.yaml"
        logger.info(f"Loading detector config from {path}")
        detector_cfg = OmegaConf.load(path)

        base_run = Path(detector_cfg.base_run)
    elif cfg.base_path is not None:
        base_run = Path(cfg.base_path)
    else:
        raise ValueError("Must specify either detector_run or base_run")

    if cfg.save_path is not None:
        # This should override the default of detector_path if set.
        save_path = Path(cfg.save_path)

    path = base_run / ".hydra" / "config.yaml"
    logger.info(f"Loading base model config from {path}")
    base_cfg = OmegaConf.load(path)

    full_computation = hydra.utils.call(base_cfg.model)
    full_model = computations.Model(computation=full_computation)
    full_params = utils.load(base_run / "model")["params"]

    if cfg.detector_path is not None:
        detector = AnomalyDetector.load(
            detector_run / "detector", full_model, full_params  # type: ignore
        )
    elif cfg.detector is not None:
        # We need to use the _partial_=True approach because the model is a dataclass,
        # and hydra would try to interpret that as a config object if we passed it
        # directly to instantiate.
        detector_factory = hydra.utils.instantiate(
            cfg.detector,
            _partial_=True,
        )
        detector = detector_factory(model=full_model, params=full_params)
    else:
        raise ValueError("Must specify either detector_path or detector")

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
            # TODO: this is specific to abstractions. Need to think more about what
            # the adversarial interface should look like in general.
            filter_maps = get_tau_maps(detector.abstraction.computation, identity_init)

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
                save_path=save_path,
            )
            if save_path:
                utils.save(finetuned_vars, save_path / "finetuned_vars", overwrite=True)

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
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
