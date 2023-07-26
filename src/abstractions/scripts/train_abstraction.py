import copy
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from abstractions.detectors.abstraction.abstraction import get_tau_maps

from abstractions.data import data
from abstractions.detectors.abstraction import AbstractionDetector, Abstraction
from abstractions.models.computations import Model
import abstractions.utils.hydra
from abstractions.utils.logger import DummyLogger, WandbLogger
from abstractions.utils import utils


CONFIG_NAME = Path(__file__).stem
abstractions.utils.hydra.setup_hydra(CONFIG_NAME)


@hydra.main(
    version_base=None, config_path=f"conf/{CONFIG_NAME}", config_name=CONFIG_NAME
)
def main(cfg: DictConfig):
    """Execute model training and evaluation loop.

    Args:
      cfg: Hydra configuration object.
    """
    if cfg.wandb:
        metrics_logger = WandbLogger(
            project_name="abstractions",
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )
    else:
        metrics_logger = DummyLogger()

    # Load the full model we want to abstract
    base_run = Path(cfg.base_run)
    base_cfg = OmegaConf.load(
        to_absolute_path(str(base_run / ".hydra" / "config.yaml"))
    )

    full_computation = hydra.utils.call(base_cfg.model)
    full_model = Model(computation=full_computation)
    full_params = utils.load(to_absolute_path(str(base_run / "model")))["params"]

    # hydra sets the OmegaConf struct flag, preventing adding new keys by default
    with open_dict(cfg):
        cfg.train_data = copy.deepcopy(base_cfg.train_data)
    # We want to train only on clean data.
    # TODO: This doesn't feel ideal, since transforms aren't necessarily backdoors
    # in general. Best way to handle this is probably to separate out backdoors
    # from any other transforms if needed.
    cfg.train_data.transforms = {}
    train_dataset = data.get_dataset(cfg.train_data)

    if cfg.val_data == "base":
        cfg.val_data = {"val": copy.deepcopy(base_cfg.val_data)}
    elif cfg.val_data == "same":
        cfg.val_data = {"val": copy.deepcopy(cfg.train_data)}
        cfg.val_data.val.train = False

    # First sample, only input without label and info.
    # Also need to add a batch dimension
    example_input = train_dataset[0][0][None]
    _, example_activations = full_model.apply(
        {"params": full_params}, example_input, return_activations=True
    )

    if "model" in cfg:
        if cfg.single_class:
            cfg.model.output_dim = 2
        else:
            cfg.model.output_dim = base_cfg.model.output_dim
        computation = hydra.utils.call(cfg.model)
        # TODO: Might want to make this configurable somehow, but it's a reasonable
        # default for now
        maps = get_tau_maps(computation)
        model = Abstraction(computation=computation, tau_maps=maps)
    else:
        model = None

    detector = AbstractionDetector(
        model=full_model,
        params=full_params,
        abstraction=model,
        size_reduction=cfg.size_reduction,
        max_batch_size=base_cfg.max_batch_size,
        output_loss_fn="single_class" if cfg.single_class else "kl",
    )

    val_datasets = {k: data.get_dataset(v) for k, v in cfg.val_data.items()}

    detector.train(
        train_dataset,
        batch_size=cfg.batch_size or base_cfg.batch_size,
        num_epochs=cfg.num_epochs or base_cfg.num_epochs,
        validation_datasets=val_datasets,
        optimizer=hydra.utils.instantiate(base_cfg.optim),
        example_input=example_activations,
        # Hydra sets the cwd to the right log dir automatically
        log_dir=".",
        check_val_every_n_epoch=1,
        loggers=[metrics_logger],
        enable_progress_bar=False,
    )

    detector.save("detector")


if __name__ == "__main__":
    main()
