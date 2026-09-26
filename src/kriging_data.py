"""Shared data layer for reconstruction — used by baselines and neural models.

Task definition (from TASK doc): the K-day history is available for *all*
nodes; only the target day t is partially observed. The job is to reconstruct
the hidden nodes' traffic on day t. Masking hides a fixed node subset across the
whole split (inductive-kriging style), reproducible from its seed.
"""
from __future__ import annotations

import hashlib

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import utils


@dataclass
class Bundle:
    """Everything a model needs for one (scenario, mask) configuration."""
    raw: np.ndarray            # (T, N) original-scale traffic, NaN = calendar gap
    scaled: np.ndarray         # (T, N) scaled, NaN→0 (per-node mean in z-space)
    scaler: utils.TrafficScaler
    A_hat: np.ndarray          # (N, N) normalized adjacency for the scenario
    A_raw: np.ndarray          # (N, N) un-normalized weighted adjacency (no self-loop)
    train: list[int]
    val: list[int]
    test: list[int]
    observed: np.ndarray       # (N,) bool, True = observed on target day
    K: int
    nodes: list[str]
    # Blend weight for the periodicity-matched level, DERIVED FROM TRAIN ROWS ONLY.
    # It is a statistic (like the scaler's mean/std), not a tuned hyper-parameter:
    # the fraction of the flat-centred residual's variance that day-of-week explains.
    # 0 => plain window mean. Every model that centres gets the SAME value, so the
    # comparison between any two centred models stays matched (used by _cd only).
    dow_weight: float = 0.0
    # Per-node statistics for the support-aware model, ALL FROM TRAIN ROWS ONLY.
    #   rho  (N,) how well a node's centred series is explained by its neighbours
    #   q_lo/q_hi (N,) the range of window levels the node actually showed in training
    # These are statistics, like the scaler's mean/std -- not learned parameters, so
    # they cannot themselves go out of distribution.
    rho: np.ndarray | None = None
    q_lo: np.ndarray | None = None
    q_hi: np.ndarray | None = None
    # component graphs for time-varying blending (C3); all leakage-safe
    components: dict | None = None   # {geo_hat, corr_hat, geo_raw, corr_raw}
    # Protocol-B history holes: (T, N) bool, True = the entry is observed.
    # None (default) = the protocol of the main experiments, where every node's
    # history is complete and only day t is hidden.
    hist_obs: np.ndarray | None = None

    @property
    def hidden(self) -> np.ndarray:
        return ~self.observed


def load_bundle(cfg: dict, scenario: str, ratio: float, strategy: str,
                seed: int, graph_override: str | None = None) -> Bundle:
    proc = Path(cfg["paths"]["processed"])
    matrix = pd.read_parquet(proc / "traffic_matrix.parquet")
    nodes = list(matrix.columns)
    raw = matrix.values.astype(float)

    splits = utils.load_json(proc / "splits.json")
    sc = splits["scenarios"][scenario]
    scaler = utils.TrafficScaler.from_dict(sc["scaler"])
    if _LEVEL_ALPHA != 1.0:
        b = min(sc["test"]) - (splits["window_K"] if _LEVEL_ONSET == "pre" else 0)
        b = max(0, b)
        raw = raw.copy()
        # ``_LEVEL_NODES`` restricts the shift to a subset of nodes. The default
        # None keeps the original behaviour, a uniform shift over every node; a
        # subset makes the shift regional, which is the case where a node's own
        # neighbours can no longer stand in for its level.
        if _LEVEL_NODES is None:
            raw[b:] *= _LEVEL_ALPHA
        else:
            raw[np.ix_(np.arange(b, raw.shape[0]), _LEVEL_NODES)] *= _LEVEL_ALPHA
    scaled = scaler.transform(raw)
    scaled = np.nan_to_num(scaled, nan=0.0)        # gaps → node mean (z=0)

    graph = graph_override or sc["graph"]
    A_hat = np.load(proc / f"{graph}_hat.npy")
    A_raw = np.load(proc / f"{graph}.npy")

    # component graphs for the time-varying model: geo + the scenario's corr graph
    # (e.g. A_mixed → A_corr, A_mixed_pre → A_corr_pre) — same train span, no leakage
    corr_name = graph.replace("A_mixed", "A_corr") if "mixed" in graph else "A_corr"
    components = {
        "geo_hat": np.load(proc / "A_geo_hat.npy"),
        "corr_hat": np.load(proc / f"{corr_name}_hat.npy"),
        "geo_raw": np.load(proc / "A_geo.npy"),
        "corr_raw": np.load(proc / f"{corr_name}.npy"),
    }

    masks = utils.load_json(proc / "masks.json")
    observed = np.array(masks[str(ratio)][strategy][str(seed)], dtype=bool)

    K = splits["window_K"]
    # Depends only on (dataset, scenario), but load_bundle is called ~20x per
    # training run by the evaluation loop, so cache it — uncached it added 15-30%
    # to every run.
    ck = (str(proc), scenario)
    if ck not in _DOW_CACHE:
        dates = pd.to_datetime(utils.load_json(proc / "dates.json")["dates"])
        _DOW_CACHE[ck] = _dow_weight(scaled, np.asarray(sc["train"]), dates, K)
    dw = _DOW_CACHE[ck]
    sk = (str(proc), scenario, graph)
    if sk not in _SUP_CACHE:
        _SUP_CACHE[sk] = _support_stats(scaled, np.asarray(sc["train"]), A_raw, K)
    rho, q_lo, q_hi = _SUP_CACHE[sk]

    return Bundle(raw=raw, scaled=scaled, scaler=scaler, A_hat=A_hat, A_raw=A_raw,
                  train=sc["train"], val=sc["val"], test=sc["test"],
                  observed=observed, K=K, nodes=nodes,
                  components=components, dow_weight=dw,
                  rho=rho, q_lo=q_lo, q_hi=q_hi,
                  hist_obs=_hist_obs_mask(*raw.shape))


_DOW_CACHE: dict = {}
_SUP_CACHE: dict = {}

# --------------------------------------------------------------------------- #
# Protocol-B history holes
# --------------------------------------------------------------------------- #
# Process-wide, set once by the runner before any bundle is loaded. A global
# rather than a parameter because load_bundle is reached through train_one and
# test_metrics, and the setting must be identical for every model in a run --
# that identity is the whole point of the comparison.
_HIST_MISS = {"rate": 0.0, "pattern": "point", "seed": 0, "block_len": 5}

# Controlled level shift for the sweep in the appendix. Multiplies the raw signal
# on the evaluation span (and the K days before it, so the history a test day
# sees is consistent) by a constant. Training rows are never touched, so a model
# trained once can be evaluated at every alpha.
_LEVEL_ALPHA = 1.0
# Node subset the shift applies to (None = every node) and how stale the hidden
# nodes' own history is allowed to be. Both default to the original protocol.
_LEVEL_NODES: np.ndarray | None = None
_STALE_DAYS = 0
# Where the shift starts. "pre" scales from K days before the test span, so every
# history a test day sees is already at the new level. "onset" starts exactly at
# the test span, so the first K test days see a history that straddles the
# transition, which is what a model actually meets on the day a shock begins.
_LEVEL_ONSET = "pre"


def set_level_shift(alpha: float, onset: str = "pre", nodes=None) -> None:
    """``nodes`` None shifts every node; an index array shifts only those."""
    global _LEVEL_ALPHA, _LEVEL_ONSET, _LEVEL_NODES
    _LEVEL_ALPHA = float(alpha)
    _LEVEL_ONSET = onset
    _LEVEL_NODES = None if nodes is None else np.asarray(nodes, dtype=int)


def set_history_staleness(days: int) -> None:
    """Blank the last ``days`` entries of every HIDDEN node's own window.

    The level a centring operation can compute is only as fresh as the node's
    most recent observation. Setting this to d means the newest value that node
    contributes to its own window is d days old, with the rest of the panel
    untouched, so freshness is varied without changing anything else.
    """
    global _STALE_DAYS
    _STALE_DAYS = int(days)
_HM_CACHE: dict = {}


def set_hist_missing(rate: float, pattern: str = "point", seed: int = 0,
                     block_len: int = 5) -> None:
    _HIST_MISS.update(rate=float(rate), pattern=pattern, seed=int(seed),
                      block_len=int(block_len))
    _HM_CACHE.clear()


def _hist_obs_mask(T: int, N: int) -> np.ndarray | None:
    """(T, N) bool, True = observed. Depends only on (T, N, rate, pattern, seed),
    so every model in a run sees the SAME holes.

    The seed is a digest of the key rather than ``hash`` of it: Python salts the
    hash of strings per process, so ``hash`` would hand a different mask to every
    invocation and the holes would not be reproducible across runs.
    """
    c = _HIST_MISS
    if c["rate"] <= 0.0:
        return None
    key = (T, N, c["rate"], c["pattern"], c["seed"], c["block_len"])
    if key in _HM_CACHE:
        return _HM_CACHE[key]
    digest = hashlib.sha256(repr(key).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big") % (2 ** 32))
    if c["pattern"] == "point":
        obs = rng.random((T, N)) >= c["rate"]
    elif c["pattern"] == "block":
        # Contiguous runs per node, mean length ``block_len`` -- GRIN/SPIN call
        # this "block missing" and it is the harder of the two.
        L = max(1, c["block_len"])
        obs = np.ones((T, N), dtype=bool)
        nb = max(1, int(round(c["rate"] * T / L)))
        for j in range(N):
            for _ in range(nb):
                ln = int(np.clip(rng.geometric(1.0 / L), 1, T))
                st = int(rng.integers(0, max(1, T - ln + 1)))
                obs[st:st + ln, j] = False
    else:
        raise ValueError(f"unknown hist pattern {c['pattern']!r}")
    # A node with no history at all makes the level undefined for EVERY method,
    # which is protocol A, not B. Guarantee each node keeps some observations.
    dead = obs.sum(0) == 0
    if dead.any():
        obs[rng.integers(0, T, size=int(dead.sum())), np.flatnonzero(dead)] = True
    _HM_CACHE[key] = obs
    return obs


def _support_stats(scaled, train_idx, A_raw, K):
    """Per-node neighbour reliability and the level range seen during training.

    ``rho``  correlation, over train days, between a node's centred value and the
             weighted mean of its neighbours' centred values. Measured: median
             0.976 on Chicago (0 nodes below 0.3) but 0.602 on airports, where
             42 of 221 nodes fall below 0.3 -- for those the neighbour anchor is
             noise, which is why it must be down-weighted per node rather than
             trusted uniformly.
    ``q_lo/q_hi``  1st/99th percentile of the node's K-day window level. The gate
             measures how far the current level sits outside this range, i.e. the
             per-node, per-day version of the shift/amplitude diagnostic.
    """
    idx = train_idx[train_idx >= K]
    N = scaled.shape[1]
    if len(idx) < 2 * K:
        return np.zeros(N), np.full(N, -1.0), np.full(N, 1.0)
    win = np.stack([scaled[t - K:t] for t in idx])
    mu = win.mean(axis=1)
    cen = np.nan_to_num(scaled[idx] - mu)
    w = A_raw / np.clip(A_raw.sum(1), 1e-6, None)[:, None]
    nb = cen @ w.T
    rho = np.zeros(N)
    for j in range(N):
        if cen[:, j].std() > 1e-9 and nb[:, j].std() > 1e-9:
            rho[j] = np.corrcoef(cen[:, j], nb[:, j])[0, 1]
    rho = np.nan_to_num(np.clip(rho, 0.0, 1.0))
    q_lo, q_hi = np.percentile(mu, 1, axis=0), np.percentile(mu, 99, axis=0)
    q_hi = np.maximum(q_hi, q_lo + 1e-3)
    return rho, q_lo, q_hi


def _dow_weight(scaled, train_idx, dates, K) -> float:
    """Fraction of the flat-centred residual's variance explained by day-of-week.

    Computed on TRAIN ROWS ONLY — no test information enters. Returns 0 when the
    signal has no weekly structure, which makes the blended level collapse back to
    the plain window mean (bikeshare: ~0.14; Chicago 'L': ~0.69).
    """
    idx = train_idx[train_idx >= K]
    if len(idx) < 2 * K:
        return 0.0
    win = np.stack([scaled[t - K:t] for t in idx])
    resid = scaled[idx] - win.mean(axis=1)
    dow = dates[idx].dayofweek.values
    means = np.stack([resid[dow == d].mean(axis=0) for d in range(7)])
    num = float(np.mean(means.var(axis=0)))
    den = float(resid.var())
    return 0.0 if den <= 0 else float(min(max(num / den, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# Torch dataset (lazy import so numpy-only baselines stay torch-free)
# --------------------------------------------------------------------------- #
def make_dataset(bundle: Bundle, target_indices: list[int]):
    """Returns raw ingredients per target day; masking is applied by the
    train/eval loop (dynamic random masks for training, fixed mask for eval)."""
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self):
            return len(target_indices)

        def __getitem__(self, i):
            t = target_indices[i]
            h = bundle.scaled[t - bundle.K:t]                        # (K, N)
            if _STALE_DAYS > 0:
                # The hidden nodes stopped reporting d days ago; every other node
                # is untouched, so only the freshness of the level changes.
                d = min(_STALE_DAYS, bundle.K)
                hid = bundle.observed == 0
                h = h.copy()
                keep = h[:bundle.K - d]
                fill = (keep[:, hid].mean(0) if bundle.K - d > 0
                        else np.zeros(int(hid.sum())))
                h[bundle.K - d:, hid] = fill[None, :]
            if bundle.hist_obs is not None:
                # Impute-then-model, applied identically to every method. Holes
                # are filled with the node's mean over the entries it DID keep in
                # this window, which makes the window mean of the filled series
                # exactly equal to the observed-only mean. Filling with 0 instead
                # would drag the level toward the node's global mean and silently
                # disable centring in precisely the regime under test.
                o = bundle.hist_obs[t - bundle.K:t]
                cnt = o.sum(0)
                m = np.where(o, h, 0.0).sum(0) / np.maximum(cnt, 1)
                m = np.where(cnt > 0, m, 0.0)
                h = np.where(o, h, m[None, :])
            hist = torch.tensor(h, dtype=torch.float32)              # (K, N)
            x_t = torch.tensor(bundle.scaled[t], dtype=torch.float32)  # (N,) scaled
            y_raw = torch.tensor(np.nan_to_num(bundle.raw[t], nan=0.0),
                                 dtype=torch.float32)                # (N,) original
            valid = torch.tensor(~np.isnan(bundle.raw[t]),
                                 dtype=torch.float32)                # (N,) has GT
            return hist, x_t, y_raw, valid

    return _DS()
