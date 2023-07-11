import copy
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

from abstractions import abstraction, data, train_abstraction, utils
from abstractions.computations import get_tau_maps, identity_init
from abstractions.mahalanobis import MahalanobisDetector
from abstractions.train_abstraction import (
    AbstractionDetector,
    kl_loss_fn,
    single_class_loss_fn,
)

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
    full_model = abstraction.Model(computation=full_computation)
    full_params = utils.load(base_run / "model")["params"]

    # TODO: this should be more robust. Maybe AnomalyDetector should have a .load()
    # classmethod? (and checkpoints would need to store the type and some hparams)
    if "size_reduction" in detector_cfg:
        # We're dealing with an AbstractionDetector
        # TODO: copied from train_abstraction.py, should refactor
        if "model" not in detector_cfg:
            with open_dict(detector_cfg):
                detector_cfg.model = copy.deepcopy(base_cfg.model)
            if detector_cfg.model._target_ not in train_abstraction.KNOWN_ARCHITECTURES:
                raise ValueError(
                    f"Model architecture {detector_cfg.model._target_} "
                    "not yet supported for size_reduction."
                )
            for field in {"hidden_dims", "channels", "dense_dims"}:
                if field in detector_cfg.model:
                    detector_cfg.model[field] = [
                        dim // detector_cfg.size_reduction
                        for dim in detector_cfg.model[field]
                    ]

        if detector_cfg.single_class:
            detector_cfg.model.output_dim = 2
        else:
            detector_cfg.model.output_dim = base_cfg.model.output_dim

        computation = hydra.utils.call(detector_cfg.model)
        # TODO: Might want to make this configurable somehow, but it's a reasonable
        # default for now
        maps = get_tau_maps(detector_cfg.model)
        model = abstraction.Abstraction(computation=computation, tau_maps=maps)

        detector = AbstractionDetector(
            model=full_model,
            params=full_params,
            abstraction=model,
            max_batch_size=base_cfg.max_batch_size,
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
            filter_maps = get_tau_maps(detector_cfg.model, identity_init)

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
