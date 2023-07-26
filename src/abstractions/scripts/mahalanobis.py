from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from abstractions.detectors.mahalanobis import MahalanobisDetector

from abstractions.data import data
from abstractions.models import computations
from abstractions.utils import utils
import abstractions.utils.hydra


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
    # Load the full model we want to abstract
    base_run = Path(cfg.base_run)
    base_cfg = OmegaConf.load(
        to_absolute_path(str(base_run / ".hydra" / "config.yaml"))
    )

    full_computation = hydra.utils.call(base_cfg.model)
    full_model = computations.Model(computation=full_computation)
    full_params = utils.load(to_absolute_path(str(base_run / "model")))["params"]

    detector = MahalanobisDetector(
        model=full_model,
        params=full_params,
        max_batch_size=cfg.max_batch_size,
    )

    train_dataset = data.get_dataset(base_cfg.train_data)

    detector.train(
        train_dataset,
        max_batches=cfg.max_batches,
        relative=cfg.relative,
        rcond=cfg.rcond,
        batch_size=cfg.batch_size,
        pbar=cfg.pbar,
    )
    detector.save("detector")


if __name__ == "__main__":
    main()
