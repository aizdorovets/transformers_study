import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):

    def __init__(self, emb_dim, hidden, dropout: float = 0.0):
        super().__init__()
        self.Bottleneck = nn.Linear(emb_dim, hidden)
        self.Restore = nn.Linear(hidden, emb_dim)

    def forward(self, x):
        bottleneck = F.relu(self.Bottleneck(x))
        return self.Restore(bottleneck)
