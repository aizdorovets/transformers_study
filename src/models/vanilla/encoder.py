import torch
import torch.nn as nn

from attention import SelfAttention
from norm import LayerNorm


class EncoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.LayerNorm = LayerNorm()

    def forward(self, x):
        attn = self.SelfAttention(x)
        intermed = self.LayerNorm(x + attn)
        ff = self.FF(intermed)
        h = self.LayerNorm(intermed + ff)