"""Adapters from the published model definitions to this paper's interface.

Only the plumbing is written here. Every module doing the modelling is imported
from ``official/``, which holds the authors' files unmodified.

Task mapping. Our sample is a fully observed K-day history plus a partially
observed target day. The published imputation models take a window with a mask,
so the window handed to them is ``[hist ; x_obs * m]`` of length K+1 whose last
step carries the node mask, and the prediction is read off that last step. That
is the same object those models were built to fill, so nothing about their
architecture has to change.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .networks import _Base
from .official.ignnk import IGNNK as _IGNNK


def _random_walk(a_raw: torch.Tensor):
    """Forward and backward random-walk supports, as IGNNK's own training script
    builds them from the adjacency."""
    a = a_raw + torch.eye(a_raw.size(0), device=a_raw.device)
    fwd = a / a.sum(1, keepdim=True).clamp_min(1e-8)
    bwd = a.t() / a.t().sum(1, keepdim=True).clamp_min(1e-8)
    return fwd, bwd


class IGNNKOfficial(_Base):
    """IGNNK exactly as published: three diffusion graph convolutions with a
    residual on the second, reconstructing the whole window."""

    # IGNNK_train.py defaults: z=100 hidden units, diffusion order K=1.
    HP = dict(lr=1e-4, batch=4)

    def __init__(self, n_nodes, K, hidden=100, order=1, **_):
        super().__init__()
        self.net = _IGNNK(K + 1, hidden, order)
        self._supports = None

    def set_graph(self, a_hat, a_raw=None):
        super().set_graph(a_hat, a_raw)
        self._supports = _random_walk(a_raw if a_raw is not None else a_hat)

    def forward(self, hist, x_obs, obs_mask):
        x = torch.cat([hist, (x_obs * obs_mask).unsqueeze(1)], dim=1)  # (B,K+1,N)
        fwd, bwd = self._supports
        return self.net(x, fwd, bwd)[:, -1]


# --------------------------------------------------------------------------- #
from .official.grin_model import GRINet as _GRINet


class GRINOfficial(_Base):
    """GRIN as published: bidirectional GRIL, per-step first-stage estimate and
    spatial decoder, autoregressive input replacement, diffusion convolutions."""

    # config/grin/la_point.yaml: d_hidden 64, d_emb 8, d_ff 64, ff_dropout 0.
    HP = dict(batch=32)

    def __init__(self, n_nodes, K, hidden=64, dropout=0.0, **_):
        super().__init__()
        self.n_nodes, self.K, self.hidden, self.dropout = n_nodes, K, hidden, dropout
        self.net = None

    def set_graph(self, a_hat, a_raw=None):
        super().set_graph(a_hat, a_raw)
        if self.net is None:
            adj = (a_raw if a_raw is not None else a_hat).detach().cpu().numpy()
            self.net = _GRINet(adj=adj, d_in=1, d_hidden=self.hidden,
                               d_ff=self.hidden, ff_dropout=self.dropout,
                               n_layers=1, kernel_size=2, decoder_order=1,
                               global_att=False, d_u=0, d_emb=8,
                               layer_norm=False, merge='mlp',
                               impute_only_holes=False).to(a_hat.device)

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        x = torch.cat([hist, (x_obs * obs_mask).unsqueeze(1)], dim=1).unsqueeze(-1)
        m = torch.cat([torch.ones(B, K, N, device=hist.device),
                       obs_mask.unsqueeze(1)], dim=1).unsqueeze(-1).bool()
        out = self.net(x, mask=m)
        imputation = out[0] if isinstance(out, (tuple, list)) else out
        return imputation[:, -1, :, 0]


# --------------------------------------------------------------------------- #
from .official.spin_model import SPINModel as _SPINModel


def _edge_index(a_raw: torch.Tensor):
    idx = (a_raw > 0).nonzero(as_tuple=False).t().contiguous()
    return idx


class SPINOfficial(_Base):
    """SPIN as published: sparse graph additive attention over (node, step)
    tokens, positional and node encodings, four layers with per-layer readout."""

    # config/imputation/spin.yaml: hidden 32, n_layers 4, eta 3, lr 8e-4,
    # batch 8 split into 2 chunks, i.e. 4 samples at a time.
    HP = dict(lr=8e-4, batch=4)

    def __init__(self, n_nodes, K, hidden=32, n_layers=4, eta=3, **_):
        super().__init__()
        self.net = _SPINModel(input_size=1, hidden_size=hidden, n_nodes=n_nodes,
                              u_size=1, output_size=1, n_layers=n_layers, eta=eta)
        self._ei = None

    def set_graph(self, a_hat, a_raw=None):
        super().set_graph(a_hat, a_raw)
        self._ei = _edge_index(a_raw if a_raw is not None else a_hat)

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        x = torch.cat([hist, (x_obs * obs_mask).unsqueeze(1)], dim=1).unsqueeze(-1)
        m = torch.cat([torch.ones(B, K, N, device=hist.device),
                       obs_mask.unsqueeze(1)], dim=1).unsqueeze(-1)
        # no exogenous covariates in this task; the encoder still supplies the
        # node embedding and the sinusoidal step encoding
        u = torch.zeros(B, K + 1, 1, device=hist.device)
        x_hat, _ = self.net(x, u, m, self._ei)
        return x_hat[:, -1, :, 0]


# --------------------------------------------------------------------------- #
from .official.satcn_model import general_satcn as _SATCN


class SATCNOfficial(_Base):
    """SATCN as published: masked principal-neighbourhood aggregation followed by
    temporal convolutions.

    Two notes on the plumbing. The masked layer wants a per-sample, per-step
    adjacency with the unknown nodes' columns zeroed, so it is built from the
    graph on the fly and only the last step carries the node mask, which is what
    this task hides. The authors' ``sadj_transform`` re-sparsifies the adjacency
    to the k nearest neighbours before each call; our graph is already built as a
    top-10 kNN graph (S6.1), so that step is a no-op here and is skipped.
    """

    # demo_metr.py: channels 128, layers 1, t_kernel 2, batch 8, lr 1e-3.
    HP = dict(lr=1e-3, batch=8)

    def __init__(self, n_nodes, K, hidden=128, layers=1, t_kernel=2, dropout=0.1, **_):
        super().__init__()
        self.hidden, self.layers, self.t_kernel, self.dropout = hidden, layers, t_kernel, dropout
        self.net = None

    def set_graph(self, a_hat, a_raw=None):
        super().set_graph(a_hat, a_raw)
        if self.net is None:
            a = a_raw if a_raw is not None else a_hat
            avg_d = {'log': torch.log(a.sum(0) + 1).mean()}
            self.net = _SATCN(avg_d, a.device, layers=self.layers,
                              t_kernel=self.t_kernel, channels=self.hidden,
                              dropout=self.dropout).to(a.device)

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        a = self.a_raw
        x = torch.cat([hist, (x_obs * obs_mask).unsqueeze(1)], dim=1)   # (B,K+1,N)
        x = x.unsqueeze(1)                                              # (B,1,T,N)
        # (B,T,N,N): columns of hidden nodes zeroed, and only on the target step
        lk_mask = a.view(1, 1, N, N).expand(B, K + 1, N, N).clone()
        lk_mask[:, -1] = a.unsqueeze(0) * obs_mask.unsqueeze(1)
        out = self.net(x, a, lk_mask)                                   # (B,1,T',N)
        return out[:, 0, -1]


# --------------------------------------------------------------------------- #
from .official.kits_lib_nn_models_kits import KITS as _KITS
from .official.grin_spatial_conv import SpatialConvOrderK as _SCK


class _KitsArgs:
    dataset_name = "custom"
    use_adj_drop = False
    use_init = True


class KITSOfficial(_Base):
    """KITS's architecture as published, called through its own ``impute``.

    Partial on purpose, and the limit is worth stating. KITS's headline
    contribution is the increment training strategy: during training it grows an
    augmented graph of virtual entries and reconnects them each epoch. That is a
    training procedure, not a module, and adopting it for one backbone would
    change the protocol every other backbone is held to. What is reproduced here
    is the network the paper defines, three diffusion convolutions with the
    reference mechanism between them; what is not reproduced is how the paper
    trains it.
    """

    # config/kits/la_point.yaml: d_hidden 64, batch 32.
    HP = dict(batch=32)

    def __init__(self, n_nodes, K, hidden=64, **_):
        super().__init__()
        self.hidden = hidden
        self.net = None

    def set_graph(self, a_hat, a_raw=None):
        super().set_graph(a_hat, a_raw)
        if self.net is None:
            a = (a_raw if a_raw is not None else a_hat)
            self.net = _KITS(adj=a.detach().cpu().numpy(), d_in=1,
                             d_hidden=self.hidden, args=_KitsArgs()).to(a.device)
        self._supp = _SCK.compute_support(self.a_raw, self.a_raw.device)

    def forward(self, hist, x_obs, obs_mask):
        B, K, N = hist.shape
        x = torch.cat([hist, (x_obs * obs_mask).unsqueeze(1)], dim=1).unsqueeze(-1)
        m = torch.cat([torch.ones(B, K, N, device=hist.device),
                       obs_mask.unsqueeze(1)], dim=1).unsqueeze(-1)
        out = self.net.impute(x, m, self._supp)          # (B,S,N,1)
        return out[:, -1, :, 0]
