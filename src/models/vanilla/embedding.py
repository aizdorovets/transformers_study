import math

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionalEmbeddingLayer(nn.Module):

    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        positions = torch.arange(max_seq_len).unsqueeze(1)
        dimensions = 2 * (torch.arange(embed_dim).div(2))
        divisor = 1 / (10_000 ** (dimensions / embed_dim)).unsqueeze(0)
        self.pe = positions * divisor
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = torch.sin(self.pe[:, 1::2])

    def forward(self, x: torch.Tensor):
        batch_size, max_len, _ = x.shape
        return self.pe[:max_len, :].repeat(batch_size, 1, 1)
