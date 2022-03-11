import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, hidden_dim, eps: float = 1e-5):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, x):
        # BS x TS x Emb
        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, keepdim=True)
        return self.weights * ((x - mu) / (sigma + self.eps)) + self.bias
