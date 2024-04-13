import torch
import torchmetrics
import lightning.pytorch as pl
import torch.nn as nn

from torchvision.models import swin_v2_b, Swin_V2_B_Weights


class SwinTransformer(pl.LightningModule):
    def __init__(self, num_classes: int = 10, learning_rate: float = 1e-3):
        super().__init__()
        model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)

        self.model = model

        self.example_input_array = torch.Tensor(1, 3, 224, 224)
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.save_hyperparameters()

    def forward(self, *args: torch.Any, **kwargs: torch.Any) -> torch.Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        tensors, targets = batch
        logits = self.model(tensors)
        loss = nn.functional.cross_entropy(logits, targets)
        preds = nn.functional.softmax(logits, dim=-1)
        accuracy = torchmetrics.functional.accuracy(
            preds, targets, task="multiclass", num_classes=self.num_classes
        )
        self.log("train/loss", loss)
        self.log("train/acc", accuracy)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        tensors, targets = batch
        logits = self.model(tensors)
        val_loss = nn.functional.cross_entropy(logits, targets)
        preds = nn.functional.softmax(logits, dim=-1)
        accuracy = torchmetrics.functional.accuracy(
            preds, targets, task="multiclass", num_classes=self.num_classes
        )
        self.log("val/loss", val_loss)
        self.log("val/acc", accuracy)
        return {"loss", val_loss}

    def test_step(self, batch, batch_idx):
        tensors, targets = batch
        logits = self.model(tensors)
        test_loss = nn.functional.cross_entropy(logits, targets)
        preds = nn.functional.softmax(logits, dim=-1)
        accuracy = torchmetrics.functional.accuracy(
            preds, targets, task="multiclass", num_classes=self.num_classes
        )
        self.log("test/loss", test_loss)
        self.log("test/acc", accuracy)

    def predict_step(self, tensors, batch_idx):
        pred = self.model(tensors)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
