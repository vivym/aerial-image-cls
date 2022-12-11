from typing import Optional, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex


class FCDenseNet(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 1000,
        learning_rate: float = 1e-3,
        max_epochs: int = 60,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.FPN(
            model_name,
            encoder_weights=None,
            classes=num_classes,
            in_channels=4,
        )

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
        )
        self.m_iou_train = MulticlassJaccardIndex(num_classes=num_classes)
        self.m_iou_val = MulticlassJaccardIndex(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _training_and_validation_step(self, batch, batch_idx: int):
        images, labels, image_paths = batch

        outputs = self.forward(images)

        preds = outputs.argmax(1)

        loss = F.cross_entropy(
            outputs, labels
        )
        loss += self.dice_loss(outputs, labels)

        if self.training:
            self.m_iou_train.update(outputs, labels)
        else:
            self.m_iou_val.update(outputs, labels)

        return image_paths, preds, loss

    def training_step(self, batch, batch_idx: int):
        _, _, loss = self._training_and_validation_step(batch, batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs):
        m_iou = self.m_iou_train.compute()

        self.log("train/m_iou", m_iou, prog_bar=True)

    def validation_step(self, batch, batch_idx: int):
        image_paths, preds, loss = self._training_and_validation_step(batch, batch)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return image_paths, preds

    def validation_epoch_end(self, outputs):
        m_iou = self.m_iou_val.compute()

        self.log("val/m_iou", m_iou, prog_bar=True)

        if self.current_epoch > 0:
            torch.save(outputs, "results/results.pth")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
