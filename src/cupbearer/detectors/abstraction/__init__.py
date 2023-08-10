from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cupbearer.data import DatasetConfig
from cupbearer.detectors.abstraction.adversarial_detector import (
    AdversarialAbstractionDetector,
)
from cupbearer.models.computations import Model
from cupbearer.utils.config_groups import config_group, register_config_group
from cupbearer.utils.optimizers import Adam, OptimizerConfig
from cupbearer.utils.utils import BaseConfig, get_object

from ..config import DetectorConfig, StoredDetector, TrainConfig
from .abstraction import Abstraction, get_default_abstraction
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
class AbstractionConfig(BaseConfig):
    cls: str = "cupbearer.detectors.abstraction.abstraction.Abstraction"
    size_reduction: int = 4
    output_loss_fn: str = "kl"
    # TODO: there should be a way of customizing the abstraction's architecture.
    # The best way to achieve this might be to have a 'ComputationConfig' class instead
    # of just 'ModelConfig', and then use a config group for that as a field here
    # or in another abstraction config class.

    def build(self, model: Model) -> Abstraction:
        abstraction_cls = get_object(self.cls)
        return get_default_abstraction(
            model,
            self.size_reduction,
            output_dim=2 if self.output_loss_fn == "single_class" else None,
            abstraction_cls=abstraction_cls,
        )


@dataclass
class AxisAlignedAbstractionConfig(AbstractionConfig):
    cls: str = "cupbearer.detectors.abstraction.abstraction.AxisAlignedAbstraction"


ABSTRACTIONS = {
    "simple": AbstractionConfig,
    "axis_aligned": AxisAlignedAbstractionConfig,
}
register_config_group(AbstractionConfig, ABSTRACTIONS)


@dataclass
class AbstractionDetectorConfig(DetectorConfig):
    abstraction: AbstractionConfig = config_group(AbstractionConfig, AbstractionConfig)
    train: AbstractionTrainConfig = field(default_factory=AbstractionTrainConfig)

    def build(self, model, params, rng, save_dir) -> AbstractionDetector:
        abstraction = self.abstraction.build(model)
        return AbstractionDetector(
            model=model,
            params=params,
            rng=rng,
            abstraction=abstraction,
            output_loss_fn=self.abstraction.output_loss_fn,
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )


@dataclass
class AdversarialAbstractionConfig(AbstractionDetectorConfig):
    load_path: Optional[Path] = None
    num_train_samples: int = 128
    num_steps: int = 1
    normal_weight: float = 0.5
    clip: bool = True
    # This is a bit hacky: this detector doesn't support training, so maybe we shouldn't
    # inherit from AbstractionConfig.
    train: None = None

    def build(self, model, params, rng, save_dir) -> AdversarialAbstractionDetector:
        if self.load_path is not None:
            helper_cfg = StoredDetector(path=self.load_path)
            detector = helper_cfg.build(model, params, rng, save_dir)
            assert isinstance(detector, AbstractionDetector)
            return AdversarialAbstractionDetector(
                model=model,
                params=params,
                rng=rng,
                abstraction=detector.abstraction,
                abstraction_state=detector.abstraction_state,
                output_loss_fn=detector.output_loss_fn,
                num_train_samples=self.num_train_samples,
                num_steps=self.num_steps,
                normal_weight=self.normal_weight,
                clip=self.clip,
                save_path=save_dir,
            )

        abstraction = self.abstraction.build(model)
        return AdversarialAbstractionDetector(
            model=model,
            params=params,
            rng=rng,
            abstraction=abstraction,
            output_loss_fn=self.abstraction.output_loss_fn,
            save_path=save_dir,
            num_train_samples=self.num_train_samples,
            num_steps=self.num_steps,
            normal_weight=self.normal_weight,
            clip=self.clip,
        )

    def _set_debug(self):
        super()._set_debug()
        self.num_train_samples = 1
        self.num_steps = 1
