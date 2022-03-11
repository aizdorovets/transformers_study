from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import SelfAttention
from embedding import EmbeddingLayer, PositionalEmbeddingLayer
from encoder import Encoder
from decoder import Decoder


class VanillaTransformer(nn.Module):

    def __init__(
        self,
        embed_dim: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        feedforward_dim: int = 2048,
        vocab_size: int = 50_000,
        max_seq_len: int = 512,
        dropout: float = 0.2,
        embedding_dropout: float = 0.1,
        pad_id: int = 0,
        device: str = 'cpu',
    ):
        """
        Args:
            embed_dim: dimensionality of embeddings.
            vocab_size: how many tokens there are in tokenizer's vocabulary.
            max_seq_len: how many tokens there might be in the longest sample.
            embedding_dropout_p: probability of dropout.
        """
        super().__init__()
        # layers
        self.Embedding = EmbeddingLayer(
            vocab_size,
            embed_dim,
            pad_id=pad_id,
        )
        self.PositionalEmbedding = PositionalEmbeddingLayer(
            embed_dim,
            max_seq_len,
        )
        self.EmbeddingDropout = nn.Dropout(p=embedding_dropout)
        self.EncoderBlock = Encoder(
            embed_dim,
            n_heads,
            feedforward_dim,
            dropout,
            n_encoder_layers,
        )
        self.DecoderBlock = Decoder(
            embed_dim,
            n_heads,
            feedforward_dim,
            dropout,
            n_decoder_layers,
        )
        # self.Decoders = nn.ModuleList([
        #     DecoderLayer() for _ in range(n_decoder_layers)
        # ])
        self.Exit = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_mask: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: torch.Tensor,
    ):
        hidden = self.Embedding(inputs) + self.PositionalEmbedding(inputs)
        hidden = self.EmbeddingDropout(hidden)
        encoder_output = self.EncoderBlock(hidden, inputs_mask)
        hidden = self.Embedding(targets) + self.PositionalEmbedding(targets)
        hidden = self.EmbeddingDropout(hidden)
        hidden = self.DecoderBlock(hidden, targets_mask, encoder_output, inputs_mask)
        logits = self.Exit(hidden)
        probas = F.softmax(logits, dim=-1)
        return probas, logits
