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
        return optimizer

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
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        del batch_idx
        x, y = batch
        logits = self(x)

        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.config.model_mlp["output_dim"]
        )

        print(f"Preds: {preds.tolist()}")
        print(f"Target: {y.tolist()}")

        return {"loss": loss, "acc": acc}
    
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
