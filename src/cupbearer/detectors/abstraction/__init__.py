from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cupbearer.data import DatasetConfig
from cupbearer.detectors.abstraction.adversarial_detector import (
    AdversarialAbstractionDetector,
)
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.optimizers import Adam, OptimizerConfig

from ..config import DetectorConfig, TrainConfig
from .abstraction import Abstraction
from .abstraction_detector import AbstractionDetector


@dataclass
class AbstractionTrainConfig(TrainConfig):
    batch_size: int = 128
    num_epochs: int = 10
    validation_datasets: dict[str, DatasetConfig] = field(default_factory=dict)
    optimizer: OptimizerConfig = config_group(OptimizerConfig, Adam)
    check_val_every_n_epoch: int = 1
    enable_progress_bar: bool = False
    max_steps: Optional[int] = None
    # TODO: should be possible to configure loggers (e.g. wandb)

    def _set_debug(self):
        super()._set_debug()
        self.batch_size = 2
        self.num_epochs = 1
        self.max_steps = 1


@dataclass
class AbstractionConfig(DetectorConfig):
    abstraction: Optional[Abstraction] = None
    size_reduction: Optional[int] = 4
    # TODO: configuring abstraction_cls currently doesn't work, looks like
    # simple_parsing just sets this to `type` instead of `Abstraction`.
    # abstraction_cls: type[Abstraction] = Abstraction
    output_loss_fn: str = "kl"
    train: AbstractionTrainConfig = field(default_factory=AbstractionTrainConfig)

    def build(self, model, params, save_dir) -> AbstractionDetector:
        return AbstractionDetector(
            model=model,
            params=params,
            abstraction=self.abstraction,
            size_reduction=self.size_reduction,
            # abstraction_cls=self.abstraction_cls,
            output_loss_fn=self.output_loss_fn,
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )


@dataclass
class AdversarialAbstractionConfig(AbstractionConfig):
    load_path: Optional[Path] = None
    num_at_once: int = 1
    num_ref_samples: int = 128
    num_steps: int = 1
    normal_weight: float = 0.5
    clip: bool = True
    # This is a bit hacky: this detector doesn't support training, so maybe we shouldn't
    # inherit from AbstractionConfig.
    train: None = None

    def build(self, model, params, save_dir) -> AdversarialAbstractionDetector:
        if self.load_path is not None:
            return AdversarialAbstractionDetector.from_detector(
                path=self.load_path / "detector",
                model=model,
                params=params,
                num_at_once=self.num_at_once,
                num_ref_samples=self.num_ref_samples,
                num_steps=self.num_steps,
                normal_weight=self.normal_weight,
                clip=self.clip,
                save_path=save_dir,
            )

        return AdversarialAbstractionDetector(
            model=model,
            params=params,
            abstraction=self.abstraction,
            size_reduction=self.size_reduction,
            # abstraction_cls=self.abstraction_cls,
            output_loss_fn=self.output_loss_fn,
            save_path=save_dir,
            num_at_once=self.num_at_once,
            num_ref_samples=self.num_ref_samples,
            num_steps=self.num_steps,
            normal_weight=self.normal_weight,
            clip=self.clip,
        )

    def _set_debug(self):
        super()._set_debug()
        self.num_at_once = 1
        self.num_ref_samples = 1
        self.num_steps = 1
