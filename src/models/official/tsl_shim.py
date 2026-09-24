"""The handful of tsl components SPIN imports, reproduced from the tsl source.

tsl itself is not installed here, and pulling it in would drag Lightning and a
data stack the rest of this repository does not use. These four classes are
copied from tsl's own files (`nn/layers/base/{embedding,dense}.py`,
`nn/blocks/encoders/mlp.py`) so SPIN's model code runs unchanged;
``PositionalEncoding`` is the standard sinusoidal encoding tsl implements.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor, nn


class StaticGraphEmbedding(nn.Module):
    """tsl's NodeEmbedding: one learned vector per node, uniform init."""

    def __init__(self, n_nodes: int, emb_size: int, **_):
        super().__init__()
        self.n_nodes, self.emb_size = int(n_nodes), int(emb_size)
        self.emb = nn.Parameter(Tensor(self.n_nodes, self.emb_size))
        self.reset_emb()

    def reset_emb(self):
        with torch.no_grad():
            bound = 1.0 / math.sqrt(self.emb.size(-1))
            self.emb.data.uniform_(-bound, bound)

    def get_emb(self):
        return self.emb

    def forward(self, expand: Optional[List] = None, node_index=None,
                nodes_first: bool = True, token_index=None):
        emb = self.get_emb()
        idx = node_index if node_index is not None else token_index
        if idx is not None:
            emb = emb[idx]
        if not nodes_first:
            emb = emb.T
        if expand is None:
            return emb
        shape = [*emb.size()]
        view = [1 if d > 0 else shape.pop(0 if nodes_first else -1) for d in expand]
        return emb.view(*view).expand(*expand)


class Dense(nn.Module):
    """tsl's Dense: linear, activation, dropout."""

    def __init__(self, input_size, output_size, activation='relu',
                 dropout=0., bias=True):
        super().__init__()
        self.affinity = nn.Linear(input_size, output_size, bias=bias)
        act = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU,
               'tanh': nn.Tanh, 'silu': nn.SiLU, 'linear': nn.Identity,
               'prelu': nn.PReLU, 'gelu': nn.GELU, 'sigmoid': nn.Sigmoid,
               'softplus': nn.Softplus, None: nn.Identity}
        self.activation = act[activation]()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        return self.dropout(self.activation(self.affinity(x)))


class MLP(nn.Module):
    """tsl's MLP: n Dense layers then an optional linear readout."""

    def __init__(self, input_size, hidden_size, output_size=None,
                 exog_size=None, n_layers=1, activation='relu', dropout=0.):
        super().__init__()
        if exog_size is not None:
            input_size += exog_size
        self.mlp = nn.Sequential(*[
            Dense(input_size=input_size if i == 0 else hidden_size,
                  output_size=hidden_size, activation=activation, dropout=dropout)
            for i in range(n_layers)])
        self.readout = nn.Linear(hidden_size, output_size) if output_size is not None else None

    def forward(self, x, u=None):
        if u is not None:
            x = torch.cat([x, u], dim=-1)
        out = self.mlp(x)
        return self.readout(out) if self.readout is not None else out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added along the step axis."""

    def __init__(self, d_model, dropout=0., max_len=5000, affinity=False,
                 batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.affinity = nn.Linear(d_model, d_model) if affinity else None
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].size(1)])
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.affinity is not None:
            x = self.affinity(x)
        # x: [..., steps, nodes, channels] or [..., steps, channels]
        steps = x.size(-3) if x.dim() >= 3 else x.size(-2)
        pe = self.pe[:steps]
        pe = pe.view(*([1] * (x.dim() - 3)), steps, 1, -1) if x.dim() >= 3 \
            else pe.view(*([1] * (x.dim() - 2)), steps, -1)
        return self.dropout(x + pe)


LayerNorm = nn.LayerNorm


# --------------------------------------------------------------------------- #
# tsl.nn.functional.sparse_softmax, copied so SPIN's attention runs unchanged.
from typing import Optional as _Opt
from torch_scatter import scatter
from torch_scatter.utils import broadcast
from torch_geometric.utils.num_nodes import maybe_num_nodes


def sparse_softmax(src: Tensor,
                   index: Optional[Tensor] = None,
                   ptr: Optional[Tensor] = None,
                   num_nodes: Optional[int] = None,
                   dim: int = -2) -> Tensor:
    r"""Extension of :func:`~torch_geometric.softmax` with index broadcasting
    to compute a sparsely evaluated softmax over multiple broadcast dimensions.

    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (Tensor, optional): The indices of elements for applying the
            softmax.
            (default: :obj:`None`)
        ptr (Tensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, i.e.,
            :obj:`max_val + 1` of :attr:`index`.
            (default: :obj:`None`)
        dim (int): The dimension on which to normalize, i.e., the edge
            dimension.
            (default: :obj:`-2`)
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        expanded_index = broadcast(index, src, dim)
        src_max = scatter(src, expanded_index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, expanded_index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-8)
