import copy
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.detectors.config import DetectorConfig
from cupbearer.scripts._shared import Classifier
from cupbearer.utils import utils
from cupbearer.utils.train import TrainConfig


class FinetuningAnomalyDetector(AnomalyDetector):
    def __init__(self, model, max_batch_size, save_path):
        super().__init__(model, max_batch_size, save_path)
        # We might as well make a copy here already, since whether we'll train this
        # detector or load weights for inference, we'll need to copy in both cases.
        self.finetuned_model = copy.deepcopy(self.model)

    @property
    def should_train_on_clean_data(self) -> bool:
        return True

    def train(
        self,
        clean_dataset,
        *,
        num_classes: int,
        train_config: TrainConfig,
    ):
        classifier = Classifier(
            self.finetuned_model,
            num_classes=num_classes,
            optim_cfg=train_config.optim,
            save_hparams=False,
        )

        # Create a DataLoader for the clean dataset
        clean_loader = train_config.get_dataloader(clean_dataset)

        # Finetune the model on the clean dataset
        trainer = train_config.get_trainer()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "You defined a `validation_step` but have no `val_dataloader`."
                    " Skipping val loop."
                ),
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
class FinetuningConfig(DetectorConfig):
    train: TrainConfig = field(default_factory=TrainConfig)

    def build(self, model, save_dir) -> FinetuningAnomalyDetector:
        return FinetuningAnomalyDetector(
            model=model,
            max_batch_size=self.train.max_batch_size,
            save_path=save_dir,
        )
