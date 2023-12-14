from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cupbearer.data import DatasetConfig
from cupbearer.models import HookedModel
from cupbearer.utils.config_groups import config_group, register_config_group
from cupbearer.utils.train import TrainConfig
from cupbearer.utils.utils import BaseConfig

from ..config import DetectorConfig
from .abstraction import (
    Abstraction,
    AutoencoderAbstraction,
    LocallyConsistentAbstraction,
)
from .abstraction_detector import AbstractionDetector


@dataclass
class AbstractionTrainConfig(TrainConfig):
    validation_datasets: dict[str, DatasetConfig] = field(default_factory=dict)


# This is all unnessarily verbose right now, it's a remnant from when we had
# robust optimization for abstractions and I experimented with some variations.
# Leaving it like this for now, but ultimately, the way to go is probably to just
# let users specify a path to a python function that gets called
# to construct the abstraction. (With get_default_abstraction being the default.)
@dataclass
class AbstractionConfig(BaseConfig, ABC):
    size_reduction: int = 4

    @abstractmethod
    def build(self, model: HookedModel) -> Abstraction:
        pass


class LocallyConsistentAbstractionConfig(AbstractionConfig):
    def build(self, model: HookedModel) -> LocallyConsistentAbstraction:
        return LocallyConsistentAbstraction.get_default(
            model,
            self.size_reduction,
        )


class AutoencoderAbstractionConfig(AbstractionConfig):
    def build(self, model: HookedModel) -> AutoencoderAbstraction:
        return AutoencoderAbstraction.get_default(
            model,
            self.size_reduction,
        )


ABSTRACTIONS = {
    "lca": LocallyConsistentAbstractionConfig,
    "autoencoder": AutoencoderAbstractionConfig,
}
register_config_group(AbstractionConfig, ABSTRACTIONS)


@dataclass
class AbstractionDetectorConfig(DetectorConfig):
    abstraction: AbstractionConfig = config_group(
        AbstractionConfig, LocallyConsistentAbstractionConfig
    )
    train: AbstractionTrainConfig = field(default_factory=AbstractionTrainConfig)

    def build(self, model, save_dir) -> AbstractionDetector:
        abstraction = self.abstraction.build(model)
        return AbstractionDetector(
            model=model,
            abstraction=abstraction,
            max_batch_size=self.train.max_batch_size,
            save_path=save_dir,
        )
