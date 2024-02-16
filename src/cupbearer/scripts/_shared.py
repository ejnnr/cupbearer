import lightning as L
import torch
from torchmetrics.classification import Accuracy

from cupbearer.models import HookedModel, ModelConfig
from cupbearer.utils.optimizers import OptimizerConfig


class Classifier(L.LightningModule):
    def __init__(
        self,
        model: ModelConfig | HookedModel,
        num_classes: int,
        optim_cfg: OptimizerConfig,
        input_shape: tuple[int, ...] | None = None,
        val_loader_names: list[str] | None = None,
        test_loader_names: list[str] | None = None,
        save_hparams: bool = True,
    ):
        super().__init__()
        if isinstance(model, HookedModel) and save_hparams:
            raise ValueError(
                "Cannot save hyperparameters when model is already instantiated. "
                "Either pass a ModelConfig or set save_hparams=False."
            )
        if save_hparams:
            self.save_hyperparameters()
        if val_loader_names is None:
            val_loader_names = []
        if test_loader_names is None:
            test_loader_names = []

        if isinstance(model, HookedModel):
            self.model = model
        elif input_shape is None:
            raise ValueError(
                "Must provide input_shape when passing a ModelConfig "
                "instead of an instantiated model."
            )
        else:
            self.model = model.build_model(input_shape=input_shape)
        self.optim_cfg = optim_cfg
        self.val_loader_names = val_loader_names
        self.test_loader_names = test_loader_names
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torch.nn.ModuleList(
            [
                Accuracy(task="multiclass", num_classes=num_classes)
                for _ in val_loader_names
            ]
        )
        self.test_accuracy = torch.nn.ModuleList(
            [
                Accuracy(task="multiclass", num_classes=num_classes)
                for _ in test_loader_names
            ]
        )

    def _shared_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.train_accuracy(logits, y)
        self.log("train/acc_step", self.train_accuracy)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits, y = self._shared_step(batch)
        name = self.test_loader_names[dataloader_idx]
        self.log(f"{name}/loss", loss)
        self.test_accuracy[dataloader_idx](logits, y)
        self.log(f"{name}/acc_step", self.test_accuracy[dataloader_idx])

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits, y = self._shared_step(batch)
        name = self.val_loader_names[dataloader_idx]
        self.log(f"{name}/loss", loss)
        self.val_accuracy[dataloader_idx](logits, y)
        self.log(f"{name}/acc_step", self.val_accuracy[dataloader_idx])

    def on_train_epoch_end(self):
        self.log("train/acc_epoch", self.train_accuracy)

    def on_test_epoch_end(self):
        for i, name in enumerate(self.test_loader_names):
            self.log(f"{name}/acc_epoch", self.test_accuracy[i])

    def on_validation_epoch_end(self):
        for i, name in enumerate(self.val_loader_names):
            self.log(f"{name}/acc_epoch", self.val_accuracy[i])

    def configure_optimizers(self):
        return self.optim_cfg.get_optimizer(self.parameters())
