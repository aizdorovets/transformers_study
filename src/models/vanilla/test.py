import matplotlib.pyplot as plt
import seaborn as sns
import torch

from embedding import PositionalEmbeddingLayer


PositionalEmbedding = PositionalEmbeddingLayer(512, 768)
x = torch.zeros((1, 512, 768))
pe = PositionalEmbedding.forward(x)
sns.heatmap(pe.squeeze(0))
plt.show()
