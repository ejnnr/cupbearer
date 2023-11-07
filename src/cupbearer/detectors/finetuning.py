import copy
from dataclasses import dataclass, field
from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.detectors.config import DetectorConfig, TrainConfig
from cupbearer.scripts.train_classifier import Classifier
from cupbearer.utils import utils
from cupbearer.utils.config_groups import config_group
from cupbearer.utils.optimizers import Adam, OptimizerConfig


class FinetuningAnomalyDetector(AnomalyDetector):
    def __init__(self, model, max_batch_size, save_path):
        super().__init__(model, max_batch_size, save_path)
        # We might as well make a copy here already, since whether we'll train this
        # detector or load weights for inference, we'll need to copy in both cases.
        self.finetuned_model = copy.deepcopy(self.model)

    def train(
        self,
        clean_dataset,
        optimizer: OptimizerConfig,
        num_classes: int,
        num_epochs: int = 10,
        batch_size: int = 128,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        classifier = Classifier(
            self.model,
            num_classes=num_classes,
            optim_cfg=optimizer,
            save_hparams=False,
        )

        # Create a DataLoader for the clean dataset
        clean_loader = DataLoader(
            dataset=clean_dataset,
            batch_size=batch_size,
        )

        # Finetune the model on the clean dataset
        trainer = L.Trainer(
            max_epochs=num_epochs,
            max_steps=max_steps or -1,
            default_root_dir=self.save_path,
        )
        trainer.fit(
            model=classifier,
            train_dataloaders=clean_loader,
        )

    def layerwise_scores(self, batch):
        raise NotImplementedError(
            "Layerwise scores don't exist for finetuning detector"
        )

    def scores(self, batch):
        inputs = utils.inputs_from_batch(batch)
        original_output = self.model(inputs)
        finetuned_output = self.finetuned_model(inputs)

        # F.kl_div requires log probabilities for the input, normal probabilities
        # are fine for the target.
        log_finetuned_p = finetuned_output.log_softmax(dim=-1)
        original_p = original_output.softmax(dim=-1)

        # This computes KL(original || finetuned), the argument order for the pytorch
        # function is swapped compared to the mathematical notation.
        # See https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        # This is the same direction of KL divergence that Redwood used in one of their
        # projects, though I don't know if they had a strong reason for it.
        # Arguably a symmetric metric would make more sense, but might not matter much.
        #
        # Also note we don't want pytorch to do any reduction, since we want to
        # return individual scores for each sample.
        kl = F.kl_div(log_finetuned_p, original_p, reduction="none").sum(-1)

        if torch.any(torch.isinf(kl)):
            # We'd get an error anyway once we compute eval metrics, but better to give
            # a more specific one here.
            raise ValueError("Infinite KL divergence")

        return kl

    def _get_trained_variables(self, saving: bool = False):
        return self.finetuned_model.state_dict()

    def _set_trained_variables(self, variables):
        self.finetuned_model.load_state_dict(variables)


@dataclass
class FinetuningTrainConfig(TrainConfig):
    optimizer: OptimizerConfig = config_group(OptimizerConfig, Adam)
    num_epochs: int = 10
    batch_size: int = 128
    max_steps: Optional[int] = None

    def setup_and_validate(self):
        super().setup_and_validate()
        if self.debug:
            self.num_epochs = 1
            self.max_steps = 1
            self.batch_size = 2


@dataclass
class FinetuningConfig(DetectorConfig):
    train: FinetuningTrainConfig = field(default_factory=FinetuningTrainConfig)

    def build(self, model, save_dir) -> FinetuningAnomalyDetector:
        return FinetuningAnomalyDetector(
            model=model,
            max_batch_size=self.max_batch_size,
            save_path=save_dir,
        )
