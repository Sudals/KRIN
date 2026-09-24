"""Tier-2 SOTA kriging / imputation baselines (faithful-in-spirit adaptations).

All conform to the common interface ``forward(hist, x_obs, obs_mask) -> (B, N)``
in scaled space, so they slot into the same train/eval harness as our models.
Each is adapted to the spatial-block kriging setting (whole nodes hidden on the
target day t; K-day history available for all nodes); the docstrings note where
the adaptation departs from the original paper.

  SATCN  Wu et al. 2021      masked multi-aggregation + temporal convolution
  GRIN   Cini et al. ICLR'22 graph-recurrent (GCN-gated GRU) + spatial decoder
  SPIN   Marisca et al. '22  inter-node sparse attention (hidden ← observed)
  KITS   ICLR'24             IGNNK backbone + learnable virtual nodes (graph gap)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConv, GraphConvStack
from .networks import _Base


# --------------------------------------------------------------------------- #
def _masked_mean(a_raw, h, obs_mask):
    """Mean of neighbour features over OBSERVED neighbours. a_raw:(N,N), h:(B,N,F)."""
    W = a_raw.unsqueeze(0) * obs_mask.unsqueeze(1)        # (B,N,N) observed columns
    denom = W.sum(-1, keepdim=True).clamp_min(1e-6)       # (B,N,1)
    return torch.einsum("bij,bjf->bif", W, h) / denom


# --------------------------------------------------------------------------- #
class SATCN(_Base):
    """Masked spatial aggregation + temporal convolution.

    Unobserved nodes are prevented from sending messages (masked aggregation),
    matching SATCN's core idea; temporal context is a 1-D conv over the window.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=3, dropout=0.1):
        super().__init__()
        self.tconv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.inp = nn.Linear(hidden + 2, hidden)
        # each layer fuses self + masked-mean(neighbours)  (PNA-lite)
        self.layers = nn.ModuleList(
            nn.Linear(2 * hidden, hidden) for _ in range(gconv_layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        seq = hist.permute(0, 2, 1).reshape(B * N, 1, K)
        temb = self.tconv(seq).squeeze(-1).view(B, N, -1)            # (B,N,H)
        h = torch.relu(self.inp(torch.cat(
            [temb, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1)], dim=-1)))
        for lin in self.layers:
            agg = _masked_mean(self.a_raw, h, obs_mask)
            h = h + self.drop(torch.relu(lin(torch.cat([h, agg], dim=-1))))
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
class GraphGRUCell(nn.Module):
    """GRU cell whose gates mix neighbour state via a GCN (GRIN-style)."""

    def __init__(self, hidden):
        super().__init__()
        self.gc_r = GraphConv(hidden + 1, hidden)
        self.gc_u = GraphConv(hidden + 1, hidden)
        self.gc_c = GraphConv(hidden + 1, hidden)

    def forward(self, x, h, a_hat):
        # x: (B,N,1) current value ; h: (B,N,H)
        xh = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.gc_r(xh, a_hat))
        u = torch.sigmoid(self.gc_u(xh, a_hat))
        c = torch.tanh(self.gc_c(torch.cat([x, r * h], dim=-1), a_hat))
        return u * h + (1 - u) * c


class GRIN(_Base):
    """Graph-recurrent imputation: a GCN-gated GRU rolls over the history, then a
    spatial decoder reconstructs the target day from the recurrent state plus the
    observed snapshot. Single-direction (forward) adaptation of GRIN."""

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.cell = GraphGRUCell(hidden)
        self.dec = GraphConvStack(hidden, gconv_layers, dropout)
        self.fuse = nn.Linear(hidden + 2, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        h = torch.zeros(B, N, self.hidden, device=hist.device)
        for t in range(K):
            h = self.cell(hist[:, t].unsqueeze(-1), h, self.a_hat)
        feat = torch.relu(self.fuse(torch.cat(
            [h, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1)], dim=-1)))
        return self.head(self.dec(feat, self.a_hat)).squeeze(-1)


# --------------------------------------------------------------------------- #
class GRINBi(GRIN):
    """GRIN with the backward pass restored.

    The official GRIN is bidirectional (BiGRIL); the adaptation above keeps only
    the forward direction, which is the largest single thing it drops. GRIN is
    also the one backbone that loses to the training-free baseline even in the
    stable condition, so this variant asks whether the two facts are connected.
    A second cell rolls the window in reverse and the two final states are
    concatenated before the spatial decoder. Everything else is unchanged, so
    the only moving part is the direction.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1):
        super().__init__(n_nodes, K, hidden, gconv_layers, dropout)
        self.cell_bwd = GraphGRUCell(hidden)
        self.fuse = nn.Linear(2 * hidden + 2, hidden)

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        hf = torch.zeros(B, N, self.hidden, device=hist.device)
        hb = torch.zeros(B, N, self.hidden, device=hist.device)
        for t in range(K):
            hf = self.cell(hist[:, t].unsqueeze(-1), hf, self.a_hat)
            hb = self.cell_bwd(hist[:, K - 1 - t].unsqueeze(-1), hb, self.a_hat)
        feat = torch.relu(self.fuse(torch.cat(
            [hf, hb, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1)], dim=-1)))
        return self.head(self.dec(feat, self.a_hat)).squeeze(-1)


# --------------------------------------------------------------------------- #
class SPIN(_Base):
    """Inter-node sparse attention: hidden nodes attend to OBSERVED nodes only.

    Captures SPIN's high-missing-rate strategy of letting unobserved positions
    gather from observed ones via attention instead of fixed graph weights. A
    temporal encoder summarizes each node's window into a token.
    """

    def __init__(self, n_nodes, K, hidden=64, heads=4, blocks=2, dropout=0.1):
        super().__init__()
        self.tenc = nn.GRU(1, hidden, batch_first=True)
        self.tok = nn.Linear(hidden + 2, hidden)
        self.blocks = nn.ModuleList(
            nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            for _ in range(blocks))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(blocks))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        seq = hist.permute(0, 2, 1).reshape(B * N, K, 1)
        temb = self.tenc(seq)[0][:, -1].view(B, N, -1)
        h = torch.relu(self.tok(torch.cat(
            [temb, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1)], dim=-1)))
        # key-padding mask: True = ignore (i.e. hidden nodes are not attended to)
        key_pad = obs_mask < 0.5                                     # (B,N)
        for attn, norm in zip(self.blocks, self.norms):
            a, _ = attn(h, h, h, key_padding_mask=key_pad, need_weights=False)
            h = norm(h + a)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
class KITS(_Base):
    """IGNNK backbone augmented with learnable virtual nodes to bridge the
    train/test graph-density gap (KITS' increment idea). Real nodes exchange
    messages with V virtual nodes via attention before the spatial readout."""

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=3, n_virtual=8, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(K + 2, hidden)
        self.gstack = GraphConvStack(hidden, gconv_layers, dropout)
        self.virtual = nn.Parameter(torch.randn(n_virtual, hidden) * 0.02)
        self.to_v = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.from_v = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        win = hist.permute(0, 2, 1)
        h = torch.relu(self.inp(torch.cat(
            [win, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1)], dim=-1)))
        h = self.gstack(h, self.a_hat)
        v = self.virtual.unsqueeze(0).expand(B, -1, -1)             # (B,V,H)
        v = v + self.to_v(v, h, h, need_weights=False)[0]          # virtual ← real
        h = self.norm(h + self.from_v(h, v, v, need_weights=False)[0])  # real ← virtual
        return self.head(h).squeeze(-1)
