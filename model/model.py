from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from base import BaseModel
from layers import PositionalEncoding, TransformerEncoderLayer
from utils import patchify


class MnistModel(BaseModel):
    chw: Tuple[int, int, int] = (1, 28, 28)
    n_patches_per_side: int = 7

    d_token: int = 128
    n_heads: int = 4
    n_encoders: int = 8
    n_classes: int = 10

    linear_mapper: nn.Conv2d
    class_token: Tensor
    positional_encoding: PositionalEncoding
    transformer_encoder: nn.TransformerEncoder
    classifier: nn.Sequential

    def __init__(self, n_classes=10):
        super().__init__()

        self.n_classes = n_classes

        patch_size = int(self.chw[1] / self.n_patches_per_side)

        # 1) Linear mapper
        d_input = int(self.chw[0] * patch_size**2)
        # self.linear_mapper = nn.Linear(self.d_input, self.d_token)
        self.linear_mapper = nn.Conv2d(
            self.chw[0], self.d_token, kernel_size=patch_size, stride=patch_size)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.d_token))

        # 3) Positinal encoding
        self.positional_encoding = PositionalEncoding(self.d_token)

        # 4) Tranformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_token, nhead=self.n_heads, activation='gelu', batch_first=True, dim_feedforward=4*self.d_token)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_encoders)

        # 5) Classification MLP (Multi-Layer Perceptron)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_token, self.n_classes),
        )

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, color_channels, height, width]``
        """

        y = self.linear_mapper(x)
        y = torch.flatten(y, start_dim=2)
        y = y.transpose(1, 2)
        # shape: [batch_size, n_patches, d_token]

        n = y.size(0)
        y = torch.cat(
            (self.class_token.unsqueeze(0).repeat(n, 1, 1), y), dim=1)
        # shape: [batch_size, n_patches+1, d_token]

        y = self.positional_encoding(y)
        # shape: [batch_size, n_patches+1, d_token]

        y = self.transformer_encoder(y)
        # shape: [batch_size, n_patches+1, d_token]

        # getting only the classification token of each batch item
        y = y[:, 0]
        # shape: [batch_size, d_token]

        y = self.classifier(y)
        # shape: [batch_size, n_classes]

        # y = nn.functional.softmax(y, dim=-1)

        return y
