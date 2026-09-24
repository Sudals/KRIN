"""Three reconstruction networks sharing one interface.

  RNNKriging  temporal-only (GRU/LSTM per node) — learned analogue of HA
  IGNNK       spatial-only diffusion graph conv over the window (static graph)
  STGNN       ours: temporal recurrence + graph propagation + mask conditioning

Inputs (scaled space):
  hist     (B, K, N)   K-day history for every node
  x_obs    (B, N)      target-day values, hidden nodes zeroed
  obs_mask (B, N)      1 = observed on the target day
Output: pred (B, N) reconstructed target-day traffic (scaled).

The normalized adjacency Â is registered as a buffer at construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import GraphConvStack


class _Base(nn.Module):
    def set_graph(self, a_hat: torch.Tensor, a_raw: torch.Tensor | None = None) -> None:
        self.register_buffer("a_hat", a_hat, persistent=False)
        if a_raw is not None:
            self.register_buffer("a_raw", a_raw, persistent=False)


# --------------------------------------------------------------------------- #
class LinearKriging(_Base):
    """Per-node affine map from the window to the target day. No graph, no
    nonlinearity. Under KRIN's centring this is NLinear with a window mean in
    place of the last value, so it isolates how much of the effect survives when
    the learned component is as simple as it can be."""

    def __init__(self, n_nodes, K, hidden=64, dropout=0.1, deep=False):
        super().__init__()
        d = K + 2                                   # window, own value, own mask
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        ) if deep else nn.Linear(d, 1)

    def forward(self, hist, x_obs, obs_mask):
        feat = torch.cat([hist.permute(0, 2, 1), x_obs.unsqueeze(-1),
                          obs_mask.unsqueeze(-1)], dim=-1)
        return self.net(feat).squeeze(-1)


# --------------------------------------------------------------------------- #
class RNNKriging(_Base):
    """Per-node recurrent encoder + readout. No spatial mixing."""

    def __init__(self, n_nodes, K, hidden=64, cell="gru", dropout=0.1):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def encode(self, hist):
        B, K, N = hist.shape
        seq = hist.permute(0, 2, 1).reshape(B * N, K, 1)
        out, _ = self.rnn(seq)
        return out[:, -1].view(B, N, -1)                    # (B, N, H)

    def forward(self, hist, x_obs, obs_mask):
        emb = self.encode(hist)
        feat = torch.cat([emb, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1)], dim=-1)
        return self.head(feat).squeeze(-1)


# --------------------------------------------------------------------------- #
class IGNNK(_Base):
    """Inductive spatial kriging: graph diffusion over the signal window.

    Faithful to the IGNNK idea (AAAI'21) — reconstruct the full node set from a
    partially-observed snapshot via stacked graph convolutions on a *static*
    graph, with no explicit temporal recurrence. The window is fed as channels.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(K + 2, hidden)                 # window + x_obs + mask
        self.gstack = GraphConvStack(hidden, gconv_layers, dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, hist, x_obs, obs_mask):
        win = hist.permute(0, 2, 1)                          # (B, N, K)
        feat = torch.cat([win, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.inp(feat))
        h = self.gstack(h, self.a_hat)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
class STGNN(_Base):
    """Ours: temporal GRU per node → graph propagation → masked readout.

    The observed target value and the observed mask enter as explicit feature
    channels, so the network can distinguish "observed 0" from "hidden", and
    propagate observed signals to hidden nodes over the latent graph.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fuse = nn.Linear(hidden + 2, hidden)            # +x_obs +mask
        self.gstack = GraphConvStack(hidden, gconv_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        seq = hist.permute(0, 2, 1).reshape(B * N, K, 1)
        temporal, _ = self.gru(seq)
        temporal = temporal[:, -1].view(B, N, -1)            # (B, N, H)
        feat = torch.cat([temporal, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
class STGNNRobustTV(_Base):
    """Regime-robust kriging with a TIME-VARYING graph (C3).

    Instead of a fixed blend ``A = α·Â_geo + β·Â_corr``, a gate reads the current
    input window (level, volatility, observed fraction) and emits per-sample
    convex weights ``α(t), β(t)``. The time-varying graph is applied to BOTH the
    same-day spatial baseline and the graph-conv stack, so the model can lean on
    geography when the (train-derived) correlation structure no longer matches the
    test regime. Leakage-safe: candidate graphs are train-only and the gate only
    sees legitimate observed inputs, never hidden ground truth.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fuse = nn.Linear(hidden + 4, hidden)
        self.convs = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(gconv_layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def set_tv_graphs(self, geo_hat, corr_hat, geo_raw, corr_raw):
        self.register_buffer("geo_hat", geo_hat, persistent=False)
        self.register_buffer("corr_hat", corr_hat, persistent=False)
        self.register_buffer("geo_raw", geo_raw, persistent=False)
        self.register_buffer("corr_raw", corr_raw, persistent=False)

    def _weights(self, hist, x_obs, obs_mask):
        ctx = torch.stack([
            hist.mean(dim=(1, 2)),                              # overall level
            hist[:, -1].mean(dim=1),                           # recent level
            (hist[:, 1:] - hist[:, :-1]).abs().mean(dim=(1, 2)),  # volatility
            obs_mask.mean(dim=1),                              # observed fraction
        ], dim=-1)                                             # (B,4)
        return torch.softmax(self.gate(ctx), dim=-1)           # (B,2): [geo, corr]

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        w = self._weights(hist, x_obs, obs_mask)               # (B,2)
        wg, wc = w[:, 0].view(B, 1, 1), w[:, 1].view(B, 1, 1)
        A_raw = wg * self.geo_raw + wc * self.corr_raw         # (B,N,N) time-varying
        A_hat = wg * self.geo_hat + wc * self.corr_hat

        # time-varying same-day spatial baseline (level cue)
        W = A_raw * obs_mask.unsqueeze(1)
        b = torch.einsum("bij,bj->bi", W, x_obs) / W.sum(-1).clamp_min(1e-6)
        recent = hist[:, -3:].mean(dim=1)

        dhist = hist[:, 1:] - hist[:, :-1]
        seq = dhist.permute(0, 2, 1).reshape(B * N, K - 1, 1)
        temporal = self.gru(seq)[0][:, -1].view(B, N, -1)
        feat = torch.cat([temporal, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1),
                          b.unsqueeze(-1), recent.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        for lin in self.convs:                                 # batched graph conv
            h = h + self.drop(torch.relu(lin(torch.einsum("bij,bjf->bif", A_hat, h))))
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
class KRIN(_Base):
    """KRIN — Kriging Regime-Invariant Normalization.

    KRIN names the OPERATION, not this class: centre each node by its own K-day
    window mean, predict, add the level back. Applied to a published backbone it is
    written ``<method> + KRIN``; the best configuration in the paper is
    ``SPIN + KRIN`` (registry name ``krinat_noa``). This class is the graph-conv
    backbone carrying the same operation, kept for the ablations.

    RevIN (ICLR'22) adapted to kriging: the principled fix for the regime-shift
    root cause.

    Each node is normalized by ITS OWN K-day window statistics (μ_i, σ_i), so the
    network only ever sees ~standardized, regime-invariant inputs; the prediction
    is de-normalized with the same (μ_i, σ_i). Since μ_i is the node's recent level
    (test-time, current regime), the output is restored to the *current* regime's
    scale — there is no frozen train-level for an OOD shift to break. The spatial
    baseline and graph conv all operate in instance-normalized space.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fuse = nn.Linear(hidden + 3, hidden)             # temporal + xobs_n + mask + baseline
        self.gstack = GraphConvStack(hidden, gconv_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def _mu(self, hist):
        """Level removed before the network sees the data. Overridden by ablations."""
        return hist.mean(dim=1)

    def _baseline(self, vals, obs_mask):
        W = self.a_raw.unsqueeze(0) * obs_mask.unsqueeze(1)
        denom = W.sum(-1).clamp_min(1e-6)
        return torch.einsum("bij,bj->bi", W, vals) / denom

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        # Mean-only RevIN: remove the per-node window LEVEL (the regime-shift culprit);
        # scale is already handled by the global per-node z-score, and dividing by the
        # instance std is unstable when a node is near-constant within the window.
        mu = self._mu(hist)                                   # (B,N) per-node window level
        hin = hist - mu.unsqueeze(1)                          # centered history (regime-invariant)
        xo_in = (x_obs - mu) * obs_mask                       # centered observed target
        b = self._baseline(xo_in, obs_mask)                   # spatial anchor (centered space)
        temporal = self._temporal(hin)                        # (B,N,H)
        feat = torch.cat([temporal, xo_in.unsqueeze(-1),
                          obs_mask.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        y_in = self.head(h).squeeze(-1)                       # centered prediction
        return y_in + mu                                      # RevIN de-normalization (add level back)

    def _temporal(self, hin):
        """Encode the centred history. Overridden by the recurrence-free variant."""
        B, K, N = hin.shape
        seq = hin.permute(0, 2, 1).reshape(B * N, K, 1)
        return self.gru(seq)[0][:, -1].view(B, N, -1)


# --------------------------------------------------------------------------- #
class STGNNRobustW(_Base):
    """STGNN-R + explicit WEEKLY seasonal anchors.

    The data shows weekly persistence (predict x(t)≈x(t-7)) is far stronger than
    daily (MAE ~11.5 vs ~19.5) — weekly seasonality dominates. Crucially, a hidden
    node's OWN history is available (only day t is hidden), so x(t-7) and x(t-14)
    are usable, level-carrying, and regime-appropriate (own recent data auto-tracks
    the current regime) → robust. We add them as explicit anchor features alongside
    the same-day spatial baseline. Dynamics still via Δ-history GRU.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fuse = nn.Linear(hidden + 6, hidden)             # +x_obs,mask,b,recent,lag7,lag14
        self.gstack = GraphConvStack(hidden, gconv_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def _baseline(self, x_obs, obs_mask):
        W = self.a_raw.unsqueeze(0) * obs_mask.unsqueeze(1)
        denom = W.sum(-1).clamp_min(1e-6)
        return torch.einsum("bij,bj->bi", W, x_obs) / denom

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        b = self._baseline(x_obs, obs_mask)
        recent = hist[:, -3:].mean(dim=1)
        lag7 = hist[:, -7] if K >= 7 else hist[:, 0]           # own value 7 days ago
        lag14 = hist[:, -14] if K >= 14 else hist[:, 0]        # 14 days ago
        dhist = hist[:, 1:] - hist[:, :-1]
        temporal = self.gru(dhist.permute(0, 2, 1).reshape(B * N, K - 1, 1))[0][:, -1].view(B, N, -1)
        feat = torch.cat([temporal, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1),
                          b.unsqueeze(-1), recent.unsqueeze(-1),
                          lag7.unsqueeze(-1), lag14.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
class STGNNRobustGP(_Base):
    """STGNN-R with a closed-form GRAPH-SOBOLEV (graph-GP) kriging anchor.

    Replaces the crude 1-hop neighbour-mean level cue with the variational
    reconstruction  argmin_f  Σ_{i∈O}(f_i-x_i)² + γ fᵀL f, whose closed form is
        f = (D_O + γL)^{-1} D_O x        (D_O = diag(observed mask), L = geo Laplacian)
    i.e. the smoothest graph interpolation consistent with the observations —
    discrete harmonic extension / graph-GP posterior mean (Matérn kernel
    (L+κ²I)^{-ν}). It is a pure LINEAR function of the observed values on the fixed
    geo graph → regime-robust, and propagates over multiple hops (the Laplacian
    inverse) so it reaches isolated hidden nodes at high mask ratios where the
    1-hop mean is blind. Dynamics still come from the Δ-history GRU.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1,
                 gamma=1.0, eps=1e-3):
        super().__init__()
        self.gamma, self.eps = gamma, eps
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fuse = nn.Linear(hidden + 4, hidden)
        self.gstack = GraphConvStack(hidden, gconv_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def _sobolev_anchor(self, x_obs, obs_mask):
        """f = (D_O + γL)^{-1} D_O x — closed-form graph-Sobolev reconstruction."""
        W = self.a_raw
        L = torch.diag(W.sum(1)) - W                          # (N,N) graph Laplacian
        N = W.shape[0]
        I = torch.eye(N, device=W.device)
        A = obs_mask.unsqueeze(-1) * I + self.gamma * (L + self.eps * I)  # (B,N,N)
        rhs = (obs_mask * x_obs).unsqueeze(-1)                # (B,N,1)
        f = torch.linalg.solve(A, rhs).squeeze(-1)            # (B,N)
        return f

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        b = self._sobolev_anchor(x_obs, obs_mask)             # robust level (closed-form)
        recent = hist[:, -3:].mean(dim=1)
        dhist = hist[:, 1:] - hist[:, :-1]
        temporal = self.gru(dhist.permute(0, 2, 1).reshape(B * N, K - 1, 1))[0][:, -1].view(B, N, -1)
        feat = torch.cat([temporal, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1),
                          b.unsqueeze(-1), recent.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        return self.head(h).squeeze(-1)


class STGNNRobustGP2(STGNNRobustGP):
    """Dual-anchor variant: feed BOTH the 1-hop neighbour mean and the closed-form
    graph-Sobolev anchor as features. Both are linear functions of observed values
    (OOD-safe), so the fuse layer can combine them adaptively — local 1-hop when
    observations are dense, multi-hop Sobolev when they are sparse — with no
    fragile gating. Best-of-both across mask ratios."""

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1,
                 gamma=1.0, eps=1e-3):
        super().__init__(n_nodes, K, hidden, gconv_layers, dropout, gamma, eps)
        self.fuse = nn.Linear(hidden + 5, hidden)             # +1 for the 1-hop anchor

    def _onehop(self, x_obs, obs_mask):
        W = self.a_raw.unsqueeze(0) * obs_mask.unsqueeze(1)
        denom = W.sum(-1).clamp_min(1e-6)
        return torch.einsum("bij,bj->bi", W, x_obs) / denom

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        b_sob = self._sobolev_anchor(x_obs, obs_mask)
        b_loc = self._onehop(x_obs, obs_mask)
        recent = hist[:, -3:].mean(dim=1)
        dhist = hist[:, 1:] - hist[:, :-1]
        temporal = self.gru(dhist.permute(0, 2, 1).reshape(B * N, K - 1, 1))[0][:, -1].view(B, N, -1)
        feat = torch.cat([temporal, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1),
                          b_sob.unsqueeze(-1), b_loc.unsqueeze(-1),
                          recent.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
class GeoAttn(nn.Module):
    """Multi-head attention restricted to graph neighbours (content-based weights
    instead of a fixed adjacency). mask:(N,N) bool — True where attention allowed."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x, mask):
        B, N, F = x.shape
        dh = F // self.h
        q = self.q(x).view(B, N, self.h, dh).transpose(1, 2)
        k = self.k(x).view(B, N, self.h, dh).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (dh ** 0.5)        # (B,h,N,N)
        scores = scores.masked_fill(~mask[None, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, F)
        return self.o(out)


class GATGRU(_Base):
    """Attention-GNN baseline (RELATED_WORK Tier-1): per-node GRU over the raw
    window + graph-attention layers over the scenario graph + masked readout."""

    def __init__(self, n_nodes, K, hidden=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.fuse = nn.Linear(hidden + 2, hidden)
        self.attn = nn.ModuleList(GeoAttn(hidden, heads) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        emb = self.gru(hist.permute(0, 2, 1).reshape(B * N, K, 1))[0][:, -1].view(B, N, -1)
        h = torch.relu(self.fuse(torch.cat(
            [emb, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1)], dim=-1)))
        mask = self.a_hat > 0
        for attn, norm in zip(self.attn, self.norm):
            h = norm(h + self.drop(torch.relu(attn(h, mask))))
        return self.head(h).squeeze(-1)


class STGNNRobustIA(_Base):
    """STGNN-R strengthened (ours++): geo-restricted ATTENTION aggregation +
    ITERATIVE refinement. Two things IGNNK structurally lacks:

    - Attention over observed geo-neighbours gives content-based aggregation
      weights (vs IGNNK's fixed diffusion), still geo-restricted → regime-robust.
    - A second pass re-injects the first-pass predictions as pseudo-observations,
      letting signal flow hidden→hidden (IGNNK is single-pass; this helps most at
      high mask ratios where observed neighbours are sparse).

    Level path stays linear/robust (same-day neighbour baseline + recent mean);
    dynamics come from the Δ-history GRU. Pure inference refinement → no extra
    regime-specific parameters. Uses geo (a_hat mask, a_raw baseline)."""

    def __init__(self, n_nodes, K, hidden=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.fuse = nn.Linear(hidden + 4, hidden)
        self.attn = nn.ModuleList(GeoAttn(hidden, heads) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _baseline(self, vals, weight_mask):
        W = self.a_raw.unsqueeze(0) * weight_mask.unsqueeze(1)
        denom = W.sum(-1).clamp_min(1e-6)
        return torch.einsum("bij,bj->bi", W, vals) / denom

    def _core(self, temporal, recent, vals, obs_mask, b, gmask):
        feat = torch.cat([temporal, vals.unsqueeze(-1), obs_mask.unsqueeze(-1),
                          b.unsqueeze(-1), recent.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        for attn, norm in zip(self.attn, self.norm):
            h = norm(h + self.drop(torch.relu(attn(h, gmask))))
        return self.head(h).squeeze(-1)

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        recent = hist[:, -3:].mean(dim=1)
        dhist = hist[:, 1:] - hist[:, :-1]
        temporal = self.gru(dhist.permute(0, 2, 1).reshape(B * N, K - 1, 1))[0][:, -1].view(B, N, -1)
        gmask = self.a_hat > 0

        # pass 1: from observed neighbours only
        b1 = self._baseline(x_obs, obs_mask)
        pred1 = self._core(temporal, recent, x_obs, obs_mask, b1, gmask)

        # pass 2: re-inject pass-1 predictions as pseudo-observations
        x_aug = x_obs * obs_mask + pred1 * (1 - obs_mask)
        full = torch.ones_like(obs_mask)
        b2 = self._baseline(x_aug, full)
        pred2 = self._core(temporal, recent, x_aug, obs_mask, b2, gmask)
        return pred2


# --------------------------------------------------------------------------- #
class STGNNRobustMR(_Base):
    """Multi-relational STGNN-R: separate graph-conv channels for geo and pcorr.

    Motivation (from the graph studies): geography is the regime-robust prior;
    partial-correlation-on-innovations (pcorr) is the best *statistical* graph
    (top in-regime, near-geo under shift). A scalar blend forces one trade-off;
    instead we give each relation its OWN conv weights (R-GCN style) so the model
    can rely on pcorr in-regime/recovery and fall back to geo under shift.

    The level anchor (same-day spatial baseline) uses the robust GEO raw graph;
    feature propagation uses both relations.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1, n_rel=2):
        super().__init__()
        self.n_rel = n_rel
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.fuse = nn.Linear(hidden + 4, hidden)
        self.layers = nn.ModuleList(
            nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_rel))
            for _ in range(gconv_layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def set_mr_graphs(self, hats, geo_raw):
        for i, h in enumerate(hats):
            self.register_buffer(f"rel_hat{i}", h, persistent=False)
        self.register_buffer("geo_raw", geo_raw, persistent=False)

    def _baseline(self, x_obs, obs_mask):
        W = self.geo_raw.unsqueeze(0) * obs_mask.unsqueeze(1)
        denom = W.sum(-1).clamp_min(1e-6)
        return torch.einsum("bij,bj->bi", W, x_obs) / denom

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        b = self._baseline(x_obs, obs_mask)
        recent = hist[:, -3:].mean(dim=1)
        dhist = hist[:, 1:] - hist[:, :-1]
        seq = dhist.permute(0, 2, 1).reshape(B * N, K - 1, 1)
        temporal = self.gru(seq)[0][:, -1].view(B, N, -1)
        feat = torch.cat([temporal, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1),
                          b.unsqueeze(-1), recent.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        hats = [getattr(self, f"rel_hat{i}") for i in range(self.n_rel)]
        for lin_rel in self.layers:
            msg = sum(lin_rel[i](torch.einsum("ij,bjf->bif", hats[i], h))
                      for i in range(self.n_rel))
            h = h + self.drop(torch.relu(msg))
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
class STGNNRobust(_Base):
    """Regime-robust kriging (C2/C3): same-day spatial baseline + temporal residual.

    Motivated by the cross-regime finding that same-day spatial message-passing is
    robust to level shift while temporal memory overfits the training regime:

      baseline b_i = weighted mean of OBSERVED neighbours' same-day values
                     (in scaled space → comparable magnitudes; tracks current level)
      residual r_i = learned correction from (level-invariant Δ-history) + graph
      output       = b_i + r_i

    The temporal branch consumes day-over-day differences (Δ-history), which are
    invariant to the absolute regime level, so it learns *dynamics*, not *level*.
    Uses the un-normalized adjacency ``a_raw`` for the baseline.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        # linear level cues (robust, IGNNK-style): x_obs, mask, baseline, recent mean
        self.fuse = nn.Linear(hidden + 4, hidden)
        self.gstack = GraphConvStack(hidden, gconv_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def _baseline(self, x_obs, obs_mask):
        """b_i = sum_j A_raw_ij * x_obs_j / sum_j A_raw_ij  over observed j."""
        W = self.a_raw.unsqueeze(0) * obs_mask.unsqueeze(1)   # (B,N,N), obs columns
        denom = W.sum(-1).clamp_min(1e-6)
        return torch.einsum("bij,bj->bi", W, x_obs) / denom   # (B,N) scaled

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        # robust LEVEL cues handled by linear paths (graceful OOD extrapolation):
        b = self._baseline(x_obs, obs_mask)                   # same-day spatial anchor
        recent = hist[:, -3:].mean(dim=1)                     # own recent level (B,N)
        # in-distribution DYNAMICS only via the recurrent path (Δ-history):
        dhist = hist[:, 1:] - hist[:, :-1]                    # level-invariant
        seq = dhist.permute(0, 2, 1).reshape(B * N, K - 1, 1)
        temporal, _ = self.gru(seq)
        temporal = temporal[:, -1].view(B, N, -1)
        feat = torch.cat([temporal, x_obs.unsqueeze(-1), obs_mask.unsqueeze(-1),
                          b.unsqueeze(-1), recent.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        return self.head(h).squeeze(-1)


# Back-compat alias: KRIN was called STGNN-RevIN through state2.
STGNNRevIN = KRIN


# --------------------------------------------------------------------------- #
# Ablation variants for the "is it just instance normalisation?" question (§6).
# --------------------------------------------------------------------------- #
class KRINNoCenter(KRIN):
    """KRIN with the level path DISABLED (mu := 0), architecture untouched.

    Isolates how much of KRIN's number comes from removing the level as opposed
    to the rest of the architecture. Note this is NOT the same as STGNN-R, which
    is a different architecture (delta-history GRU + explicit linear level cues).
    """

    def _mu(self, hist):
        return torch.zeros_like(hist[:, 0])


class KRINStd(KRIN):
    """KRIN with FULL RevIN — instance mean AND standard deviation.

    Under a demand collapse a node's within-window standard deviation can become
    small, and the de-normalisation multiplies the prediction back up by it. This
    variant exists so that the effect can be measured rather than assumed.
    """

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        mu = hist.mean(dim=1)
        sd = hist.std(dim=1).clamp_min(self.eps)              # the unstable part
        hin = (hist - mu.unsqueeze(1)) / sd.unsqueeze(1)
        xo_in = ((x_obs - mu) / sd) * obs_mask
        b = self._baseline(xo_in, obs_mask)
        seq = hin.permute(0, 2, 1).reshape(B * N, K, 1)
        temporal = self.gru(seq)[0][:, -1].view(B, N, -1)
        feat = torch.cat([temporal, xo_in.unsqueeze(-1),
                          obs_mask.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        return self.head(h).squeeze(-1) * sd + mu


class Centered(_Base):
    """Wrap ANY model in KRIN's mean-only centre/restore, unchanged otherwise.

    This answers whether KRIN's advantage is simply a normalisation that any
    published method could adopt. The wrapper applies the identical operation
    KRIN uses -- centre the history and the observed target day by each
    node's own K-day window mean, run the inner model, add the level back -- so
    the only difference that remains between ``<model>_c`` and KRIN is the
    architecture, in particular whether the spatial baseline is itself computed
    in centred space.
    """

    def __init__(self, inner: nn.Module, K: int = 14, use_dow: bool = False):
        super().__init__()
        self.inner = inner
        self.use_dow = bool(use_dow)
        self.dow_weight = 0.0
        self.dow_idx = [j for j in range(K) if (K - j) % 7 == 0]

    def set_level_prior(self, w: float) -> None:
        self.dow_weight = float(w) if self.use_dow else 0.0

    def set_graph(self, a_hat, a_raw=None):
        super().set_graph(a_hat, a_raw)
        self.inner.set_graph(a_hat, a_raw)

    def forward(self, hist, x_obs, obs_mask):
        mu = hist.mean(dim=1)                                 # (B,N)
        if self.dow_weight > 0.0 and self.dow_idx:            # same blend our model uses
            dow = hist[:, self.dow_idx].mean(dim=1)
            mu = (1.0 - self.dow_weight) * mu + self.dow_weight * dow
        out = self.inner(hist - mu.unsqueeze(1),
                         (x_obs - mu) * obs_mask, obs_mask)
        return out + mu


# --------------------------------------------------------------------------- #
class CenteredLast(Centered):
    """Centre by the LAST history value instead of the window mean.

    This is what NLinear subtracts. In forecasting the nearest timestep is the
    best estimate of the level; the claim here is that reconstruction wants a
    stable recent level rather than one day's noise, and this arm measures the
    difference instead of asserting it.
    """

    def forward(self, hist, x_obs, obs_mask):
        mu = hist[:, -1]                                      # (B,N)
        out = self.inner(hist - mu.unsqueeze(1),
                         (x_obs - mu) * obs_mask, obs_mask)
        return out + mu


class CenteredStd(Centered):
    """Full RevIN: remove mean AND divide by the window standard deviation.

    Kept as a measured arm rather than an assertion. The window of a node can go
    nearly constant during a demand collapse, the instance std goes to zero, and
    the de-normalisation multiplies the error back up. eps follows RevIN's 1e-5;
    a larger clamp would hide the very failure being measured.
    """

    def forward(self, hist, x_obs, obs_mask):
        mu = hist.mean(dim=1)
        sd = hist.std(dim=1) + 1e-5
        out = self.inner((hist - mu.unsqueeze(1)) / sd.unsqueeze(1),
                         ((x_obs - mu) / sd) * obs_mask, obs_mask)
        return out * sd + mu


# --------------------------------------------------------------------------- #
class KRINv2(KRIN):
    """KRIN with the centring matched to the signal's period, and no recurrence.

    Two independent changes, both motivated by measurements rather than search:

    (a) **Periodicity-matched level.** The flat K-day window mean leaves a large
        day-of-week component in the residual the network then has to re-learn
        (measured: 39.6% of residual variance on airports, 69.0% on Chicago,
        50.2% on the subway). Blending the flat mean with the mean of the SAME
        weekday inside the window lowers the shock-period residual RMSE on three
        of four datasets. The blend weight is a FIXED constant, never learned —
        putting a learned parameter on the level path is exactly what this paper
        argues against.

    (b) **No recurrence.** The GRU is 43.2% of KRIN's parameters (12,864 of
        29,761), and ``ignnk_c`` matches KRIN with 17,793 parameters and no
        recurrence at all. Once the history is centred it carries level-invariant
        deviations, so a linear projection of the window may be enough.

    ``dow_weight=0`` recovers the original centring; ``recurrent=True`` recovers
    the GRU, so the two changes can be ablated separately.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1,
                 eps=1e-4, use_dow=True, recurrent=False):
        super().__init__(n_nodes, K, hidden=hidden, gconv_layers=gconv_layers,
                         dropout=dropout, eps=eps)
        self.use_dow = bool(use_dow)
        self.dow_weight = 0.0          # set from the bundle via set_level_prior
        self.recurrent = bool(recurrent)
        # indices of the same weekday as day t inside the window: t-7, t-14, ...
        self.dow_idx = [j for j in range(K) if (K - j) % 7 == 0]
        if not self.recurrent:
            del self.gru
            self.proj = nn.Linear(K, hidden)

    def set_level_prior(self, w: float) -> None:
        """Receive the train-derived blend weight. Ignored if this variant opts out."""
        self.dow_weight = float(w) if self.use_dow else 0.0

    def _mu(self, hist):
        flat = hist.mean(dim=1)
        if self.dow_weight == 0.0 or not self.dow_idx:
            return flat
        dow = hist[:, self.dow_idx].mean(dim=1)
        return (1.0 - self.dow_weight) * flat + self.dow_weight * dow

    def _temporal(self, hin):
        if self.recurrent:
            return super()._temporal(hin)
        return torch.relu(self.proj(hin.permute(0, 2, 1)))     # (B,N,K) -> (B,N,H)


# --------------------------------------------------------------------------- #
class KRINAnchored(KRIN):
    """KRIN where the SPATIAL anchor is also a parameter-free additive path.

    KRIN already routes the temporal level ``mu`` around the network. But the
    spatial estimate still goes *through* it: the neighbour-weighted centred mean
    ``b`` enters as one scalar among the fuse layer's inputs, where it is diluted.
    This variant completes the same idea on the spatial axis --

        x_hat = mu + b + g(.)

    -- so the network only ever predicts the residual left after both the node's
    own recent level and its observed neighbours have been accounted for. ``b`` is
    still passed as a feature so the network can modulate it.

    A wrapped baseline CANNOT do this: ``Centered`` only gets to centre the inputs,
    it cannot split the host model's spatial aggregation into an additive path.
    This is therefore the one structural claim KRIN can make that wrapping cannot
    reproduce -- and the centring stays the plain window mean, so the existing
    ``*_c`` runs are already the matched comparison and no baseline needs re-running.
    """

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        mu = self._mu(hist)
        hin = hist - mu.unsqueeze(1)
        xo_in = (x_obs - mu) * obs_mask
        b = self._baseline(xo_in, obs_mask)
        temporal = self._temporal(hin)
        feat = torch.cat([temporal, xo_in.unsqueeze(-1),
                          obs_mask.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        return self.head(h).squeeze(-1) + b + mu      # residual on top of BOTH paths


# --------------------------------------------------------------------------- #
class KRINSupport(KRIN):
    """Support-aware kriging: the network is trusted only where training covered.

    The paper's finding is not "centre the inputs" but the sharper statement that
    **a learned mapping stops being useful exactly where it is most needed** -- once
    the demand level leaves the range training covered. The remedy that follows is
    not a better architecture but a model that *knows when to stop relying on
    itself*.

        x_hat = mu + rho_i * b + (1 - gamma) * g_theta(.)

    Three of the four terms carry no learned parameters:

    * ``mu``      the node's own K-day level (removed before the network, added back)
    * ``rho_i*b`` the neighbour anchor, weighted by how well that node's neighbours
                  explained it *during training*. On airports 42 of 221 nodes score
                  below 0.3, and for them the anchor is switched off automatically;
                  on Chicago no node scores below 0.87 so it stays fully on.
    * ``gamma``   how far the current window level sits outside the range the node
                  showed in training, in units of that range. This is the per-node,
                  per-day form of the paper's shift/amplitude diagnostic, so the
                  quantity that *predicts* the failure is the same one the model
                  *acts on*.

    Why gamma must not be learned: this project already tried a learned gate
    (``STGNNRobustTV``) and it made the shock condition WORSE (12.00 vs 9.86),
    because the gate is itself an input->output mapping and its own inputs go out
    of distribution under the shock. A gate built from training quantiles cannot
    fail that way -- there is nothing in it to mis-generalise.

    In-support (gamma=0) this is an ordinary graph kriging network, so it should not
    give up the accuracy KRIN currently loses to SPIN on bikeshare. Out of support
    (gamma->1) it degrades into the parameter-free anchor rather than collapsing.
    """

    def set_support_stats(self, rho, q_lo, q_hi) -> None:
        self.register_buffer("rho", rho, persistent=False)
        self.register_buffer("q_lo", q_lo, persistent=False)
        self.register_buffer("q_hi", q_hi, persistent=False)

    def forward(self, hist, x_obs, obs_mask):
        mu = self._mu(hist)                                   # (B,N)
        hin = hist - mu.unsqueeze(1)
        xo_in = (x_obs - mu) * obs_mask
        b = self._baseline(xo_in, obs_mask)
        temporal = self._temporal(hin)
        feat = torch.cat([temporal, xo_in.unsqueeze(-1),
                          obs_mask.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        resid = self.head(h).squeeze(-1)

        # how far outside the training range of levels is this node, right now?
        span = (self.q_hi - self.q_lo).clamp_min(1e-3)
        out = torch.maximum(self.q_lo - mu, mu - self.q_hi).clamp_min(0.0)
        gamma = (out / span).clamp(0.0, 1.0)                  # (B,N), no parameters

        return mu + self.rho * b + (1.0 - gamma) * resid


# --------------------------------------------------------------------------- #
class KRINAttn(KRIN):
    """Centring + the aggregator that centring benefits most + a spatial anchor.

    Which models gained most from the centring tells us what to build. On airports
    the gains were SPIN -75%, GRU -64%, SATCN -50%, IGNNK -21%: the more expressive
    the learned aggregator, the more it had been losing to the level shift, and the
    more it recovers once the level is routed around it. IGNNK gained least because
    it is nearly linear and was already partly robust.

    So once centring removes the out-of-support problem, expressiveness pays, and
    the most expressive aggregator here is attention over the observed nodes --
    which is why ``spin_c`` beats KRIN, whose aggregation is a fixed-weight graph
    convolution.

    This model takes that aggregator, and adds the one thing wrapping SPIN cannot
    give it: an explicit parameter-free spatial anchor in centred space, weighted
    per node by how well that node's neighbours explained it during training.

        x_hat = mu + rho_i * b + Attn(centred features)

    Attention is restricted to observed nodes via a key-padding mask, so hidden
    nodes never attend to each other.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1,
                 eps=1e-4, heads=4, blocks=2, use_anchor=True, use_centring=True):
        super().__init__(n_nodes, K, hidden=hidden, gconv_layers=gconv_layers,
                         dropout=dropout, eps=eps)
        del self.gstack
        self.use_anchor = bool(use_anchor)     # ablation: drop the spatial anchor
        self.use_centring = bool(use_centring) # ablation: mu := 0
        if not self.use_anchor:                # b is no longer an input feature
            self.fuse = nn.Linear(hidden + 2, hidden)
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            for _ in range(blocks))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(blocks))

    def set_support_stats(self, rho, q_lo, q_hi) -> None:
        self.register_buffer("rho", rho, persistent=False)

    def forward(self, hist, x_obs, obs_mask):
        mu = self._mu(hist) if self.use_centring else torch.zeros_like(hist[:, 0])
        hin = hist - mu.unsqueeze(1)
        xo_in = (x_obs - mu) * obs_mask
        b = self._baseline(xo_in, obs_mask)
        temporal = self._temporal(hin)
        feats = [temporal, xo_in.unsqueeze(-1), obs_mask.unsqueeze(-1)]
        if self.use_anchor:
            feats.append(b.unsqueeze(-1))
        h = torch.relu(self.fuse(torch.cat(feats, dim=-1)))
        key_pad = obs_mask < 0.5                       # hidden nodes are not attended to
        for a, norm in zip(self.attn, self.norms):
            h = norm(h + a(h, h, h, key_padding_mask=key_pad, need_weights=False)[0])
        out = self.head(h).squeeze(-1)
        if self.use_anchor:
            rho = getattr(self, "rho", None)
            out = out + (b if rho is None else rho * b)
        return out + mu


# --------------------------------------------------------------------------- #
class KRINTemporalProbe(KRIN):
    """Diagnostic: how much does the temporal encoder actually contribute?

    ``mode='gru'``  the current encoder
    ``mode='lin'``  one linear map of the centred window (no recurrence)
    ``mode='none'`` no temporal information at all -- the network sees only the
                    same-day observed values, the mask and the spatial anchor

    Centring makes the window sum to zero, so it carries K-1 degrees of freedom
    rather than K. Before designing an encoder around that structure it is worth
    knowing whether the encoder is earning its 43% share of the parameters at all.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1,
                 eps=1e-4, mode="gru"):
        super().__init__(n_nodes, K, hidden=hidden, gconv_layers=gconv_layers,
                         dropout=dropout, eps=eps)
        self.mode = mode
        if mode != "gru":
            del self.gru
        if mode == "lin":
            self.proj = nn.Linear(K, hidden)
        if mode == "none":
            self.fuse = nn.Linear(3, hidden)

    def _temporal(self, hin):
        if self.mode == "gru":
            return super()._temporal(hin)
        if self.mode == "lin":
            return torch.relu(self.proj(hin.permute(0, 2, 1)))
        B, K, N = hin.shape
        return hin.new_zeros(B, N, 0)          # mode == "none"


# --------------------------------------------------------------------------- #
class KRINHom(KRIN):
    """Location by subtraction, scale by homogeneity.

    Centring fixes the FIRST moment: subtract each node's own recent level and add
    it back, so the network never sees the shifted level. But a second moment moves
    too. Measured on the shock split, the standard deviation of the centred
    deviations relative to training is 2.63x on airports (per-node 1.13x to 3.94x),
    1.25x on Chicago, 1.23x on the subway and only 1.10x on bikeshare -- the same
    dataset where centring itself buys nothing.

    RevIN's answer to that second moment is to divide by the instance standard
    deviation. In this setting that is fatal: during a collapse a node's window can
    be nearly constant, sigma -> 0, and the de-normalisation explodes (measured:
    MAE 1e15 on airports 2020, while the same model is fine in 2021 when levels
    have recovered).

    The alternative is to make the network **positively homogeneous** instead of
    normalising:

        f(alpha * c) = alpha * f(c)   for all alpha > 0

    A network of bias-free linear maps and ReLU satisfies this exactly, because
    ReLU(alpha x) = alpha ReLU(x) and W(alpha x) = alpha W x. Deviations twice as
    large as anything seen in training are then extrapolated correctly *by
    construction*, with no division anywhere and so nothing to diverge.

    Two consequences for the implementation:
      * every Linear drops its bias, and only ReLU is used (no LayerNorm, which
        divides and would break homogeneity as surely as RevIN does);
      * the observed-mask cannot be concatenated as a feature -- a 0/1 channel does
        not scale with alpha. It enters multiplicatively instead, as the pair
        (h * m, h * (1-m)), which keeps the same information and stays homogeneous.

    So the level is removed by subtraction and the scale is absorbed structurally.
    Neither is a learned component, and neither can go out of distribution.
    """

    def __init__(self, n_nodes, K, hidden=64, gconv_layers=2, dropout=0.1, eps=1e-4):
        super().__init__(n_nodes, K, hidden=hidden, gconv_layers=gconv_layers,
                         dropout=dropout, eps=eps)
        # rebuild every learned part without biases
        self.gru = None
        self.tproj = nn.Linear(K, hidden, bias=False)      # GRU ~ linear here anyway
        self.fuse = nn.Linear(hidden + 2, hidden, bias=False)
        self.gstack = GraphConvStack(hidden, gconv_layers, dropout, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1, bias=False))

    def _temporal(self, hin):
        return torch.relu(self.tproj(hin.permute(0, 2, 1)))

    def forward(self, hist, x_obs, obs_mask):
        mu = self._mu(hist)
        hin = hist - mu.unsqueeze(1)
        xo_in = (x_obs - mu) * obs_mask
        temporal = self._temporal(hin)                       # (B,N,H)
        # mask enters multiplicatively so the whole map stays 1-homogeneous in the
        # centred signal; concatenating the 0/1 channel would not scale with alpha
        feat = torch.cat([temporal,
                          (xo_in * obs_mask).unsqueeze(-1),
                          (xo_in * (1.0 - obs_mask)).unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fuse(feat))
        h = self.gstack(h, self.a_hat)
        return self.head(h).squeeze(-1) + mu
