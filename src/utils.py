"""Shared utilities: config, seeding, scaling, masking, metrics, geometry.

Kept deliberately small and dependency-light so every pipeline stage
(``data_prep`` → ``build_graphs`` → ``splits`` → ``baselines`` → ``train`` →
``evaluate``) can lean on the same primitives.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------- #
# Config & reproducibility
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preferred: str = "cuda"):
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances (km) for arrays of lat/lon in degrees."""
    lat_r = np.radians(lat)[:, None]
    lon_r = np.radians(lon)[:, None]
    dlat = lat_r - lat_r.T
    dlon = lon_r - lon_r.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------------------------------- #
# Scaling — log1p + per-node z-score, fit on TRAIN rows only (leakage guard)
# --------------------------------------------------------------------------- #
@dataclass
class TrafficScaler:
    log1p: bool = True
    zscore: bool = True
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "TrafficScaler":
        """``x_train`` is (T_train, N); NaNs ignored in statistics."""
        z = np.log1p(x_train) if self.log1p else x_train.astype(float)
        if self.zscore:
            self.mean_ = np.nanmean(z, axis=0)
            std = np.nanstd(z, axis=0)
            self.std_ = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = np.log1p(x) if self.log1p else x.astype(float)
        if self.zscore:
            z = (z - self.mean_) / self.std_
        return z

    def inverse(self, z: np.ndarray) -> np.ndarray:
        x = z
        if self.zscore:
            x = x * self.std_ + self.mean_
        if self.log1p:
            x = np.expm1(x)
        return np.clip(x, 0, None)

    def to_dict(self) -> dict:
        return {
            "log1p": self.log1p,
            "zscore": self.zscore,
            "mean_": None if self.mean_ is None else self.mean_.tolist(),
            "std_": None if self.std_ is None else self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrafficScaler":
        s = cls(log1p=d["log1p"], zscore=d["zscore"])
        s.mean_ = None if d["mean_"] is None else np.asarray(d["mean_"])
        s.std_ = None if d["std_"] is None else np.asarray(d["std_"])
        return s


# --------------------------------------------------------------------------- #
# Spatial masking — hide whole nodes on a target day (reproducible by seed)
# --------------------------------------------------------------------------- #
def spatial_mask(
    n_nodes: int,
    ratio: float,
    seed: int,
    strategy: str = "random",
    hub_flags: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean ``observed`` mask of length ``n_nodes``.

    ``True`` = observed (kept), ``False`` = hidden (to reconstruct).
    ``ratio`` is the fraction of nodes hidden. ``hub_stratified`` hides the
    same ratio within hubs and non-hubs separately so hub coverage is matched.
    """
    rng = np.random.default_rng(seed)
    observed = np.ones(n_nodes, dtype=bool)

    if strategy == "random":
        n_hide = int(round(ratio * n_nodes))
        hidden = rng.choice(n_nodes, size=n_hide, replace=False)
        observed[hidden] = False
    elif strategy == "hub_stratified":
        if hub_flags is None:
            raise ValueError("hub_stratified requires hub_flags")
        for group in (np.where(hub_flags)[0], np.where(~hub_flags)[0]):
            if len(group) == 0:
                continue
            n_hide = int(round(ratio * len(group)))
            hidden = rng.choice(group, size=n_hide, replace=False)
            observed[hidden] = False
    else:
        raise ValueError(f"unknown masking strategy: {strategy}")
    return observed


# --------------------------------------------------------------------------- #
# Metrics — evaluated on MASKED (hidden) nodes, in ORIGINAL scale
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred, hidden_mask) -> dict:
    """Reconstruction metrics on hidden nodes, in original scale.

    Inputs are 2-D arrays ``(samples × nodes)`` and a boolean hidden mask of the
    same shape. Returns both **micro** (pool all hidden cells; hub-dominated) and
    **macro** (per-node error, then average; treats airports equally) variants,
    plus **NMAE%** = MAE as a percentage of mean actual traffic. MAE/RMSE are in
    flights/day; sMAPE and NMAE are percentages. R² is micro (note: inflated by the
    huge between-airport variance — interpret with care).
    """
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = np.asarray(hidden_mask, bool)
    if yt.ndim == 1:                       # be forgiving; treat as one "node block"
        yt, yp, m = yt[None, :], yp[None, :], m[None, :]
    valid = m & ~np.isnan(yt) & ~np.isnan(yp)
    nan_keys = ("MAE", "RMSE", "sMAPE", "R2", "NMAE",
                "MAE_macro", "RMSE_macro", "sMAPE_macro", "NMAE_macro", "n")
    if not valid.any():
        return {k: float("nan") for k in nan_keys} | {"n": 0}

    err = yp - yt
    ae = np.abs(err)
    smape_cell = np.where((np.abs(yt) + np.abs(yp)) > 0,
                          ae / ((np.abs(yt) + np.abs(yp)) / 2.0), np.nan) * 100.0

    # ---- micro: pool all valid hidden cells -------------------------------- #
    fv = valid
    ytf, aef = yt[fv], ae[fv]
    mae = float(aef.mean())
    rmse = float(np.sqrt((err[fv] ** 2).mean()))
    smape = float(np.nanmean(smape_cell[fv]))
    mean_true = float(np.abs(ytf).mean())
    nmae = float(mae / mean_true * 100.0) if mean_true > 0 else float("nan")
    ss_res = float((err[fv] ** 2).sum())
    ss_tot = float(((ytf - ytf.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # ---- macro: per-node error, then average over nodes -------------------- #
    aen = np.where(valid, ae, np.nan)
    sen = np.where(valid, err ** 2, np.nan)
    ytn = np.where(valid, yt, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mae_j = np.nanmean(aen, axis=0)
            rmse_j = np.sqrt(np.nanmean(sen, axis=0))
            smape_j = np.nanmean(np.where(valid, smape_cell, np.nan), axis=0)
            mean_j = np.nanmean(np.abs(ytn), axis=0)
    ok = ~np.isnan(mae_j)
    mae_macro = float(np.nanmean(mae_j[ok]))
    rmse_macro = float(np.nanmean(rmse_j[ok]))
    smape_macro = float(np.nanmean(smape_j[ok]))
    nz = ok & (mean_j > 0)
    nmae_macro = float(np.nanmean(mae_j[nz] / mean_j[nz] * 100.0)) if nz.any() else float("nan")

    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2, "NMAE": nmae,
            "MAE_macro": mae_macro, "RMSE_macro": rmse_macro,
            "sMAPE_macro": smape_macro, "NMAE_macro": nmae_macro,
            "n": int(fv.sum())}


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
def regime_of_year(year: int, regimes: dict) -> str:
    for name, (lo, hi) in regimes.items():
        if lo <= year <= hi:
            return name
    return "unknown"


def years_in_regime(regime: str, regimes: dict) -> list[int]:
    lo, hi = regimes[regime]
    return list(range(lo, hi + 1))


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
