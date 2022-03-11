import torch
import torch.nn as nn

from attention import SelfAttention
from feedforward import PositionwiseFeedForward
from norm import LayerNorm


class Decoder(nn.Module):

    def __init__(
        self,
        embed_dim: int = 512,
        n_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        n_encoder_layers: int = 6,
    ):
        super().__init__()
        self.DecoderLayers = nn.ModuleList([
            DecoderLayer(
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
        enc_out: torch.Tensor,
        enc_out_mask: torch.Tensor,
    ):
        bs, ts, *_ = inputs.shape
        inputs_mask = torch.tril(torch.ones(bs, 1, ts, ts), diagonal=0)
        enc_out_mask = enc_out_mask[:, None, None, :]  # BS x None x None x TS
        for decoder in self.DecoderLayers:
            hidden = decoder(inputs, enc_out, inputs_mask, enc_out_mask)
        return hidden


class DecoderLayer(nn.Module):

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
        self.CrossAttention = SelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=0.1,
        )
        self.LayerNorm_1 = LayerNorm(hidden_dim=embed_dim)
        self.LayerNorm_2 = LayerNorm(hidden_dim=embed_dim)
        self.LayerNorm_3 = LayerNorm(hidden_dim=embed_dim)
        self.FF = PositionwiseFeedForward(embed_dim, ff_hidden)
        self.Dropout = nn.Dropout(p=dropout)

    def forward(self, x, enc_out, input_mask, enc_mask):
        self_attention = self.SelfAttention(x, x, x, input_mask)
        x = self.LayerNorm_1(x + self_attention)
        x = self.Dropout(x)
        cross_attention = self.CrossAttention(x, enc_out, enc_out, enc_mask)
        x = self.LayerNorm_2(x + cross_attention)
        x = self.Dropout(x)
        x = self.LayerNorm_3(x + self.FF(x))
        x = self.Dropout(x)
        return x
