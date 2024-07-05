import copy
import warnings
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F

from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.scripts._shared import Classifier
from cupbearer.utils import inputs_from_batch


class FinetuningAnomalyDetector(AnomalyDetector):
    def set_model(self, model):
        super().set_model(model)
        # We might as well make a copy here already, since whether we'll train this
        # detector or load weights for inference, we'll need to copy in both cases.
        self.finetuned_model = copy.deepcopy(self.model)

    def _train(
        self,
        trusted_dataloader,
        untrusted_dataloader,
        save_path: Path | str,
        *,
        num_classes: int,
        lr: float = 1e-3,
        **trainer_kwargs,
    ):
        if trusted_dataloader is None:
            raise ValueError("Finetuning detector requires trusted training data.")
        classifier = Classifier(
            self.finetuned_model,
            num_classes=num_classes,
            lr=lr,
            save_hparams=False,
        )

        # Finetune the model on the clean dataset
        trainer = L.Trainer(default_root_dir=save_path, **trainer_kwargs)
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
                train_dataloaders=trusted_dataloader,
            )

    def _compute_layerwise_scores(self, batch):
        inputs = inputs_from_batch(batch)
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

        return {"all": kl}

    def _get_trained_variables(self):
        return self.finetuned_model.state_dict()

    def _set_trained_variables(self, variables):
        self.finetuned_model.load_state_dict(variables)
