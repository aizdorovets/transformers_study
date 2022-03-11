import torch
import torch.nn.functional as F
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = self.embed_dim // self.n_heads
        if self.head_dim * self.n_heads != self.embed_dim:
            raise ValueError(
                f"Embed_dim ({self.embed_dim}) should be divisible by n_heads ({self.n_heads})"
            )
        if dropout:
            self.Dropout = nn.Dropout(p=dropout)
            self.dropout = True
        else:
            self.dropout = False
        self.scale = self.head_dim ** (1/2)

        self.W_Query = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.W_Key = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.W_Value = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.W_Projection = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.shape[0]
        # Split embeddings into heads
        queries = queries.view(batch_size, -1, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, -1, self.n_heads, self.head_dim)
        values = values.view(batch_size, -1, self.n_heads, self.head_dim)
        # Project
        # BS x TS x Heads x Head_dim -> BS x Heads x TS x Head_dim
        queries = self.W_Query(queries).permute(0, 2, 1, 3)
        keys = self.W_Key(keys).permute(0, 2, 1, 3)
        values = self.W_Value(values).permute(0, 2, 1, 3)

        attn = self.scaled_dot_product_attention(queries, keys, values, mask, self.dropout)
        # BS x Heads x TS x Head_dim -> BS x Query_TS x Emb_dim
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, -1, self.n_heads * self.head_dim)
        attention = self.W_Projection(attn)
        return attention

    def scaled_dot_product_attention(self, queries, keys, values, mask=None, dropout=False):
        qk = queries @ torch.transpose(keys, -2, -1)
        weights = qk / self.scale
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e20)
        attention = F.softmax(weights, dim=-1)
        if dropout:
            attention = self.Dropout(attention)
        return attention @ values
