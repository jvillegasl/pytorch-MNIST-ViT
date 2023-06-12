import math
import torch
import torch.nn as nn
from torch import Tensor


class MultiheadAttention(nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads

        # key, query, value projections
        self.key = nn.Linear(n_embd, n_embd*n_heads)
        self.query = nn.Linear(n_embd, n_embd*n_heads)
        self.value = nn.Linear(n_embd, n_embd*n_heads)

        # output projection
        self.proj = nn.Linear(n_embd*n_heads, n_embd)

    def forward(self, x):
        B, L, F = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, L, F, self.n_heads).transpose(
            1, 3)  # (B, nh, L, F)
        q = self.query(x).view(B, L, F, self.n_heads).transpose(
            1, 3)  # (B, nh, L, F)
        v = self.value(x).view(B, L, F, self.n_heads).transpose(
            1, 3)  # (B, nh, L, F)

        # attention (B, nh, L, F) x (B, nh, F, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v  # (B, nh, L, L) x (B, nh, L, F) -> (B, nh, L, F)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, L, F*self.n_heads)

        return self.proj(y)


class TransformerEncoderLayer(nn.Module):
    norm1: nn.LayerNorm
    mha: MultiheadAttention
    norm2: nn.LayerNorm
    mlp: nn.Sequential

    def __init__(
            self,
            d_token: int = 100,
            n_heads: int = 4,
            layer_norm_eps: float = 1e-5
    ):

        super().__init__()

        # assert d_token % n_heads == 0, 'd_token must be divisible by n_heads'

        self.norm1 = nn.LayerNorm(d_token, eps=layer_norm_eps)

        self.mha = MultiheadAttention(n_embd=d_token, n_heads=n_heads)

        self.norm2 = nn.LayerNorm(d_token, eps=layer_norm_eps)

        self.mlp = nn.Sequential(
            nn.Linear(d_token, 4 * d_token),
            nn.GELU(),
            nn.Linear(4 * d_token, d_token)
        )

    def forward(self, x: Tensor):

        t = self.norm1(x)
        y = x + self.mha(t)

        t = self.norm2(y)
        y = y + self.mlp(t)

        return y
