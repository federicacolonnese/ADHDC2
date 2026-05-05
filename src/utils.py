import torch
from matplotlib import pyplot as plt
import numpy as np


def plot_similarity_matrix(X):
    """
    X: torch.Tensor di forma (N, D)
    """
    X_norm = X / X.norm(dim=1, keepdim=True)
    sim_matrix = torch.mm(X_norm, X_norm.t()).cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(sim_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cosine similarity')
    plt.title('Similarity Matrix')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.tight_layout()
    plt.show()

    return sim_matrix
