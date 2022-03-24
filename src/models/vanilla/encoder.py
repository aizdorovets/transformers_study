import torch
import torch.nn as nn

from attention import SelfAttention
from feedforward import PositionwiseFeedForward
from norm import LayerNorm


class Encoder(nn.Module):

    def __init__(
        self,
        embed_dim: int = 512,
        n_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        n_encoder_layers: int = 6,
    ):
        super().__init__()
        self.EncoderLayers = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                ff_hidden=feedforward_dim,
                dropout=dropout,
            )
            for _ in range(n_encoder_layers)
        ])

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_mask: torch.Tensor,
    ):
        inputs_mask = inputs_mask[:, None, None, :]  # BS x None x None x TS
        for encoder in self.EncoderLayers:
            hidden = encoder(inputs, mask=inputs_mask)
        return hidden


class EncoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ff_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.SelfAttention = SelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=0.1,
        )
        self.LayerNorm_1 = LayerNorm(hidden_dim=embed_dim)
        self.LayerNorm_2 = LayerNorm(hidden_dim=embed_dim)
        self.FF = PositionwiseFeedForward(embed_dim, ff_hidden)
        self.Relu = nn.ReLU()
        self.Dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        attention = self.SelfAttention(x, x, x, mask)
        x = self.LayerNorm_1(x + attention)
        x = self.Dropout(x)
        x = self.LayerNorm_2(x + self.FF(x))
        x = self.Relu(x)
        x = self.Dropout(x)
        return x
