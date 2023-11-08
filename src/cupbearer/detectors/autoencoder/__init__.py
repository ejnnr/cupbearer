from dataclasses import dataclass, field
from typing import Optional

from cupbearer.data import DatasetConfig
from cupbearer.models import HookedModel
from cupbearer.utils.config_groups import config_group, register_config_group
from cupbearer.utils.optimizers import Adam, OptimizerConfig
from cupbearer.utils.utils import BaseConfig

from ..config import DetectorConfig, TrainConfig
from .autoencoder import ActivationAutoencoder, get_default_autoencoder
from .autoencoder_detector import AutoencoderDetector


@dataclass
class AutoencoderTrainConfig(TrainConfig):
    batch_size: int = 128
    num_epochs: int = 10
    validation_datasets: dict[str, DatasetConfig] = field(default_factory=dict)
    optimizer: OptimizerConfig = config_group(OptimizerConfig, Adam)
    check_val_every_n_epoch: int = 1
    enable_progress_bar: bool = False
    max_steps: Optional[int] = None
    # TODO: should be possible to configure loggers (e.g. wandb)

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.batch_size = 2
            self.num_epochs = 1
            self.max_steps = 1


# This is all unnessarily verbose right now, because I've just copied autoencoders
@dataclass
class AutoencoderConfig(BaseConfig):
    latent_dim: int = 32

    def build(self, model: HookedModel) -> ActivationAutoencoder:
        return get_default_autoencoder(
            model,
            self.latent_dim,
        )


AUTOENCODERS = {
    "simple": AutoencoderConfig,
}
register_config_group(AutoencoderConfig, AUTOENCODERS)


@dataclass
class AutoencoderDetectorConfig(DetectorConfig):
    autoencoder: AutoencoderConfig = config_group(AutoencoderConfig, AutoencoderConfig)
    train: AutoencoderTrainConfig = field(default_factory=AutoencoderTrainConfig)

    def build(self, model, save_dir) -> AutoencoderDetector:
        autoencoder = self.autoencoder.build(model)
        return AutoencoderDetector(
            model=model,
            autoencoder=autoencoder,
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )
