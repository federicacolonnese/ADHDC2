import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBundlerGlobal(nn.Module):
    """
    Graph module (GLOBAL adjacency) - no Linear/Conv on HVs
    Input : (B,C,H)
    Output: bundle (B,H) + A (C,C)
    """
    def __init__(self, C, steps=1, symmetric=True, self_loops=True):
        super().__init__()
        self.C = C
        self.steps = steps
        self.symmetric = symmetric
        self.self_loops = self_loops

        # Learnable adjacency logits (global, shared across samples)
        self.A_logits = nn.Parameter(torch.ones(C, C))

    def get_A(self, device):
        A = self.A_logits

        if self.self_loops:
            A = A + torch.eye(self.C, device=device)

        A = F.softmax(A, dim=-1)  # row-stochastic

        if self.symmetric:
            A = 0.5 * (A + A.t())

        return A

    def forward(self, x):  # x: (B,C,H)
        A = F.sigmoid(self.get_A(x.device))  # (C,C)

        z = x
        for _ in range(self.steps):
            # message passing: (B,C,H) = (C,C) @ (B,C,H)
            z = torch.einsum('cj,bjh->bch', A, z)

        bundle = z.sum(dim=1)  # (B,H)
        return bundle, A