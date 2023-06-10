import torch.nn as nn
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
    norm1: nn.LayerNorm
    mha: nn.MultiheadAttention
    norm2: nn.LayerNorm
    mlp: nn.Sequential

    def __init__(
            self,
            d_token: int = 100,
            n_heads: int = 4,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-5
    ):

        super().__init__()

        assert d_token % n_heads == 0, 'd_token must be completely divisible by n_heads'

        self.norm1 = nn.LayerNorm(d_token, eps=layer_norm_eps)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(d_token, eps=layer_norm_eps)

        self.mlp = nn.Sequential(
            nn.Linear(d_token, 4 * d_token),
            nn.GELU(),
            nn.Linear(4 * d_token, d_token)
        )

    def forward(self, x: Tensor):

        t = self.norm1(x)
        y = x + self.mha(t, t, t)[0]

        t = self.norm2(y)
        y = y + self.mlp(t)

        return y
