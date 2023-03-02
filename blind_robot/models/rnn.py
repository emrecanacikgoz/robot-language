from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy


class RNN(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rnn = nn.RNN(
            input_size=config.model_rnn["input_size"],
            hidden_size=config.model_rnn["hidden_size"],
            num_layers=config.model_rnn["num_layers"],
            nonlinearity="tanh",
            bias=config.model_rnn["bias"],
            batch_first=True,
            dropout=config.model_rnn["dropout"],
            bidirectional=config.model_rnn["bidirectional"],
        )

        self.fc = nn.Linear(
            config.model_rnn["hidden_size"],
            config.model_rnn["output_dim"]
        )

    def forward(self, idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # set initial hidden and cell states
        h0 = torch.zeros(
            self.config.model_rnn["num_layers"],
            idx.size(0),
            self.config.model_rnn["hidden_size"]
        ).to(device)

        # forward rnn
        out, h_n = self.rnn(idx, h0)

        # decode the hidden state of the last time step
        logits = self.fc(out)
        output = F.log_softmax(logits, dim=1)

        return output

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
