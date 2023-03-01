from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy


class MLP(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(
            config.model_mlp["input_dim"], config.model_mlp["hidden_dim"]
        )
        self.fc2 = nn.Linear(
            config.model_mlp["hidden_dim"], config.model_mlp["hidden_dim"]
        )
        self.fc3 = nn.Linear(
            config.model_mlp["hidden_dim"], config.model_mlp["output_dim"]
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.model_mlp["dropout"])

        # init weights
        self.apply(self._init_weights)

    def forward(self, idx):
        x = idx.view(idx.shape[0], -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        return out

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=float(self.config.data["lr"])
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor = self.config.data["lr_factor"],
            patience = self.config.data["lr_patience"],
            cooldown = self.config.data["lr_cooldown"],
            min_lr = float(self.config.data["min_lr"]),
            verbose = True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": self.config.trainer["check_val_every_n_epoch"]
            },
        }


    def training_step(self, batch, batch_idx):
        del batch_idx

        x, y = batch
        logits = self(x)

        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.config.model_mlp["output_dim"]
        )

        return {"loss": loss, "acc": acc}

    def training_epoch_end(self, outputs):
        loss = torch.stack([output["loss"] for output in outputs]).mean()
        acc = torch.stack([output["acc"] for output in outputs]).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc",   acc, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        del batch_idx
        x, y = batch
        logits = self(x)

        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.config.model_mlp["output_dim"]
        )

        return {"loss": loss, "acc": acc, "y":y, "preds": preds}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([output["loss"] for output in outputs]).mean()
        acc = torch.stack([output["acc"] for output in outputs]).mean()
        y = torch.stack([output["y"] for output in outputs]).view(-1)
        preds = torch.stack([output["preds"] for output in outputs]).view(-1)

        print(f"\nPreds: {preds.tolist()}")
        print(f"Target: {y.tolist()}")

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc",   acc, on_step=False, on_epoch=True)
