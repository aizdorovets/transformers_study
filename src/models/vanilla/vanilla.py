import torch
import torch.nn as nn

from embedding import EmbeddingLayer, PositionalEmbeddingLayer


class VanillaTransformer:

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        max_seq_len: int,
        embedding_dropout_p: float,
    ):
        """
        Args:
            embed_dim: dimensionality of embeddings.
            vocab_size: how many tokens there are in tokenizer's vocabulary.
            max_seq_len: how many tokens there might be in the longest sample.
            embedding_dropout_p: probability of dropout.
        """
        #
        # layers
        self.Embedding = EmbeddingLayer(
            vocab_size,
            embed_dim,
        )
        self.PositionalEmbedding = PositionalEmbeddingLayer(
            max_seq_len,
            embed_dim,
        )
        self.EmbeddingDropout = nn.Dropout(p=embedding_dropout_p)
        raise NotImplementedError

    def __call__(self):
        self.forward()

    def forward(
        self,
        inputs: torch.Tensor,
        outputs,
    ):
        embeddings = self._embed(inputs)  # BS x TS x Dim


    def _embed(
        self,
        inputs,
    ):
        emb = self.Embedding(inputs) + self.PositionalEmbedding(inputs)
        emb = self.EmbeddingDropout(emb)
        return emb