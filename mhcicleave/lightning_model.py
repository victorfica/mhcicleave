from typing import Any

import torch
import lightning.pytorch as pl
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import AUROC, MatthewsCorrCoef, Accuracy, ConfusionMatrix, ROC
from torchmetrics import MetricCollection, MetricTracker

pl.seed_everything(42)

class CleavageLightningModule(pl.LightningModule):
    def __init__(self, model, lr=0.001, step_size=3, gamma=0.1):
        """
        Un LightningModule genérico para clasificación binaria.

        Args:
            model (nn.Module): La arquitectura de red neuronal (ej. MLP, CNN).
            lr (float): Tasa de aprendizaje inicial.
            step_size (int): Periodo del StepLR scheduler.
            gamma (float): Factor de reducción del StepLR scheduler.
        """

        super().__init__()
        self.save_hyperparameters()  # guarda hiperparámetros en checkpoint

        # model and loss func
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()

        # def metrics
        self.conf = ConfusionMatrix(task="binary")
        self.roc = ROC(task="binary")
        metrics = MetricCollection([AUROC(task="binary"), Accuracy(task="binary"), MatthewsCorrCoef(task="binary")])

        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Define tracker over the collection to easy keep track of the metrics over multiple epochs
        self.train_tracker = MetricTracker(self.train_metrics)
        self.valid_tracker = MetricTracker(self.valid_metrics)
        self.test_tracker = MetricTracker(self.test_metrics)

    def forward(self, x):
        """
        x: tensor [batch_size, input_size]
        Retorna logits de tamaño [batch_size, 1]
        """
        return self.model(x)

    # shared step used in training, validation
    def _shared_steps(self, batch):
        """
        batch["encoded_sequences"]:  [batch_size, input_size]
        batch["sequence_labels"]:    [batch_size] (flotantes para BCE)
        """

        inputs = batch[0]
        labels = batch[1]

        logits = self(inputs).squeeze(-1)  # [batch_size]
        loss = self.loss_fn(logits, labels.float())

        pred_labels = torch.sigmoid(logits)
        return loss, labels, pred_labels

    def training_step(self, batch, batch_idx):
        # run shared step
        loss, labels, pred_labels = self._shared_steps(batch)

        self.train_tracker.update(pred_labels, labels)
        # logging loss and metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, pred_labels = self._shared_steps(batch)

        self.valid_tracker.update(pred_labels, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.logger.experiment.add_scalars('loss', {'valid': loss},self.global_step)

        return loss

    def test_step(self, batch, batch_idx):
        loss, labels, pred_labels = self._shared_steps(batch)

        self.roc(pred_labels, labels)
        self.conf(pred_labels, labels)

        self.test_tracker.update(pred_labels, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_start(self):
        self.train_tracker.increment()

    def on_validation_epoch_start(self):
        self.valid_tracker.increment()

    def on_test_epoch_start(self):
        self.test_tracker.increment()

    def on_train_epoch_end(self):
        epoch_metrics = self.train_tracker.compute()
        self.log_dict(epoch_metrics, logger=True)

    def on_validation_epoch_end(self):
        epoch_metrics = self.valid_tracker.compute()
        self.log_dict(epoch_metrics, logger=True)

    def on_test_epoch_end(self):
        epoch_metrics = self.test_tracker.compute()
        self.log_dict(epoch_metrics, logger=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            },
        }
    # changed batch[0] to batch in predict_step
    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        """
        Generates predictions for a batch of data.
        Applies sigmoid to logits to get probabilities.
        """
        logits = self(batch[0])
        preds = torch.sigmoid(logits)
        return preds
