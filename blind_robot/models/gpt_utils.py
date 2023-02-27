import math

import torch
from torch.nn import functional as F
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.model_gpt["n_embd"], bias=config.model_gpt["bias"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.model_gpt["n_embd"], bias=config.model_gpt["bias"])
        self.feedforward = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.feedforward(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.model_gpt["n_embd"] % config.model_gpt["n_head"] == 0, (
            f"Cannot parallelize since embedding size [{config.model_gpt['n_embd']}] is"
            f" not divisable by number of heads [{config.model_gpt['n_head']}]"
        )

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.model_gpt["n_embd"],
            3 * config.model_gpt["n_embd"],
            bias=config.model_gpt["bias"],
        )
        # output projection
        self.c_proj = nn.Linear(
            config.model_gpt["n_embd"],
            config.model_gpt["n_embd"],
            bias=config.model_gpt["bias"],
        )

        # regularization
        self.attn_dropout = nn.Dropout(config.model_gpt["dropout"])
        self.resid_dropout = nn.Dropout(config.model_gpt["dropout"])
        self.n_head = config.model_gpt["n_head"]
        self.n_embd = config.model_gpt["n_embd"]
        self.dropout = config.model_gpt["dropout"]

        # flash attention make GPU go brrrrr but support is only in PyTorch nightly
        # and still a bit scary
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and self.dropout == 0.0
        )
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention atm needs PyTorch"
                " nightly and dropout=0.0"
            )
            # causal mask to ensure that attention is only applied to the left in the
            # input sequence
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(
                        config.model_gpt["block_size"], config.model_gpt["block_size"]
                    )
                ).view(
                    1, 1, config.model_gpt["block_size"], config.model_gpt["block_size"]
                ),
            )

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to
        # be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend:
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )

        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model_gpt["feedforward"] == "fc":
            self.c_fc = nn.Linear(
                config.model_gpt["n_embd"],
                4 * config.model_gpt["n_embd"],
                bias=config.model_gpt["bias"],
            )
            self.c_proj = nn.Linear(
                4 * config.model_gpt["n_embd"],
                config.model_gpt["n_embd"],
                bias=config.model_gpt["bias"],
            )
        elif config.model_gpt["feedforward"] == "cnn":
            self.c_fc = Conv1D(
                config.model_gpt["n_embd"], 4 * config.model_gpt["n_embd"]
            )
            self.c_proj = Conv1D(
                4 * config.model_gpt["n_embd"], config.model_gpt["n_embd"]
            )
        else:
            raise NotImplementedError(
                "Other mlp connections are not implemented; supports only fc (default)"
                " and cnn1d!"
            )
        if config.model_gpt["activation"] == "gelu":
            self.activation = nn.GELU()
        elif config.model_gpt["activation"] == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(
                "Other activations are not implemented; supports only gelu (default)"
                " and relu!"
            )
        self.dropout = nn.Dropout(config.model_gpt["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias.
    PyTorch doesn't support simply bias=False
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Conv1D(nn.Module):
    """
    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L94
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
