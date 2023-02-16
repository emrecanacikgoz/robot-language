import torch
import math
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from blind_robot.model_utils import Block, LayerNorm

class gpt(LightningModule):
    def __init__(self, config):
        super().__init__()

        assert config.model["vocab_size"] is not None, "Vocab Size is not Specified"
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wpe  = nn.Embedding(config.model["block_size"], config.model["n_embd"]),
            drop = nn.Dropout(config.model["dropout"]),
            h    = nn.ModuleList([Block(config) for _ in range(config.model["n_layer"])]),
            ln_f = LayerNorm(config.model["n_embd"], bias=config.model["bias"]),
        ))
        self.lm_head = nn.Linear(config.model["n_embd"], config.model["vocab_size"], bias=config.model["bias"])

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.model["n_layer"]))
        print("number of parameters: %.2fM" % (self._get_num_params()/1e6,))

    def forward(self, idx, targets=None):
        device = self.config.trainer["accelerator"]
        b, t, e = idx.size()
        
        assert t <= self.config.model["block_size"], f"Cannot forward sequence of length {t}, block size is only {self.config.model['block_size']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        x = self.transformer.drop(idx + pos_emb)
        print(f"IDX: {idx.shape}")
        print(f"pos: {pos.shape}")
        print(f"pos_emb: {pos_emb.shape}")
        print(f"x: {x.shape}")
        for block in self.transformer.h:
            x = block(x)
        print(f"Decoder Output: {x.shape}")
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        print("Training Step")
        return logits

    def validation_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        self.log('val_loss', logits)