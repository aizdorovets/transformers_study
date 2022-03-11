import math

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_id)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        print(x)
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionalEmbeddingLayer(nn.Module):

    def __init__(self, embed_dim, max_seq_len):
        super().__init__()
        positions = torch.arange(max_seq_len).unsqueeze(1)
        dimensions = 2 * (torch.arange(embed_dim).div(2))
        divisor = 1 / (10_000 ** (dimensions / embed_dim)).unsqueeze(0)
        self.pe = positions * divisor
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])

    def forward(self, x: torch.Tensor):
        max_len = x.size(1)
        pe = self.pe[:max_len, :]
        pe.requires_grad = False
        return pe
