import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from embedding import PositionalEmbeddingLayer


PositionalEmbedding = PositionalEmbeddingLayer(512, 768)
# x = torch.zeros((1, 512, 768))
# pe = PositionalEmbedding.forward(x)
# sns.heatmap(pe.squeeze(0).detach().numpy())
# plt.show()

plt.figure(figsize=(15, 5))
pe = PositionalEmbeddingLayer(100, 20)
y = torch.zeros(1, 100, 20) + pe.forward(torch.zeros(1, 100, 20))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])
plt.show()