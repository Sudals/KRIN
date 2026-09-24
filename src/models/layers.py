"""Graph convolution over a precomputed normalized adjacency Â."""
from __future__ import annotations

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """Single GCN layer: H' = Â H W  (Â shared as a registered buffer)."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F) ; a_hat: (N, N)
        return self.lin(torch.einsum("ij,bjf->bif", a_hat, x))


class GraphConvStack(nn.Module):
    """Residual stack of GCN layers with ReLU + dropout."""

    def __init__(self, dim: int, layers: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.convs = nn.ModuleList(GraphConv(dim, dim, bias=bias) for _ in range(layers))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + self.drop(self.act(conv(x, a_hat)))
        return x
