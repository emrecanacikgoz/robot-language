import math

from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F

from blind_robot.models.gpt_utils import Block
from blind_robot.models.gpt_utils import LayerNorm


class GPT(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # pylint: disable=use-dict-literal
        self.transformer = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(
                    config.model_gpt["block_size"], config.model_gpt["n_embd"]
                ),
                drop=nn.Dropout(config.model_gpt["dropout"]),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.model_gpt["n_layer"])]
                ),
                ln_f=LayerNorm(
                    config.model_gpt["n_embd"], bias=config.model_gpt["bias"]
                ),
            )
        )

        # init weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.model_gpt["n_layer"])
                )
        # pylint: disable=C0209
        print("number of parameters: %.2fM" % (self._get_num_params() / 1e6,))

    def forward(self, idx, targets=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _, t, _ = idx.size()
        assert t <= self.config.model_gpt["block_size"], (
            f"Cannot forward sequence of length {t}, block size is only"
            f" {self.config.model_gpt['block_size']}"
        )

        # apply positional embedding
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(idx + pos_emb)

        # decoder
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # loss
        if self.config.model_gpt["loss"] == "softmax":
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)), targets.view(-1), ignore_index=0
            )
        elif self.config.model_gpt["loss"] == "mse":
            loss = F.mse_loss(
                x, targets, size_average=None, reduce=None, reduction="mean"
            )
        else:
            raise NotImplementedError(
                "Only Cross Entropy (classification) and Mean Square Error (regression)"
                " losses are supported!"
            )

        return x, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=float(self.config.data["lr"])
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        del batch_idx
        x, y = batch
        _, loss = self(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        x, y = batch
        _, loss = self(x, y)
        self.log("val_loss", loss)
        return loss
