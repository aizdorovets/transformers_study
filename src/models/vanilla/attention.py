import torch
import torch.nn.functional as F
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = self.embed_dim // self.n_heads
        if self.head_dim * self.n_heads != self.embed_dim:
            raise ValueError(
                f"Embed_dim ({self.embed_dim}) should be divisible by n_heads ({self.n_heads})"
            )

        self.W_Query = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.W_Key = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.W_Value = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.W_Projection = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        # Split embeddings into heads
        x = x.view(batch_size, -1, self.n_heads, self.head_dim)
        # Project
        # BS x TS x Heads x Head_dim -> BS x Heads x TS x Head_dim
        queries = self.W_Query(x).permute(0, 2, 1, 3)
        keys = self.W_Key(x).permute(0, 2, 1, 3)
        values = self.W_Value(x).permute(0, 2, 1, 3)

        attn = self.scaled_dot_product_attention(queries, keys, values, mask)
        # BS x Heads x TS x Head_dim -> BS x Query_TS x Emb_dim
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, -1, self.n_heads * self.head_dim)
        attention = self.W_Projection(attn)
        return attention

    def scaled_dot_product_attention(self, queries, keys, values, mask):
        qk = queries @ torch.transpose(keys, 2, 3)
        weights = qk / (keys.shape[-1] ** (1/2))
        if mask is not None:
            weights.masked_fill(mask == 0, -1e20)
        attention = F.softmax(weights, dim=-1) @ values
        return attention
