"""Phase 2 — Latent airport dependency graphs.

Constructs three adjacency families, all leakage-safe:

  A_geo     gaussian kernel over haversine distance        (static, no leakage)
  A_corr    Pearson/Spearman co-variation of traffic        (TRAIN span only)
  A_mixed   alpha * A_geo + beta * A_corr                    (normalized blend)

For the regime-aware experiments (C2/C3) we additionally emit one correlation
graph per regime (``A_corr_<regime>``) computed strictly within that regime's
days — so when a regime is used as the training set the graph is leakage-free.

Every saved adjacency comes with its symmetric-normalized form
``Â = D^-1/2 (A + I) D^-1/2`` for direct use in graph convolutions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import utils


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def normalize01(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    np.fill_diagonal(a, 0.0)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else a


def sparsify_topk(a: np.ndarray, k: int) -> np.ndarray:
    """Keep each node's top-k strongest edges, then symmetrize (union)."""
    n = a.shape[0]
    out = np.zeros_like(a)
    if k >= n - 1:
        out = a.copy()
    else:
        for i in range(n):
            idx = np.argpartition(a[i], -k)[-k:]
            out[i, idx] = a[i, idx]
    out = np.maximum(out, out.T)            # union → symmetric
    np.fill_diagonal(out, 0.0)
    return out


def normalized_adjacency(a: np.ndarray) -> np.ndarray:
    """Â = D^-1/2 (A + I) D^-1/2."""
    a = a + np.eye(a.shape[0])
    deg = a.sum(axis=1)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    return dinv[:, None] * a * dinv[None, :]


def geo_graph(lat, lon, cfg) -> np.ndarray:
    d = utils.haversine_matrix(lat, lon)
    off = d[~np.eye(len(d), dtype=bool)]
    sigma = cfg["graph"]["geo"]["sigma"] or off.std()
    a = np.exp(-(d ** 2) / (sigma ** 2))
    a[a < cfg["graph"]["geo"]["epsilon"]] = 0.0
    a = sparsify_topk(a, cfg["graph"]["geo"]["top_k"])
    return normalize01(a), float(sigma)


def corr_graph(values: pd.DataFrame, cfg) -> np.ndarray:
    """``values`` is (T_span × N) raw traffic; correlated on log1p signal."""
    z = np.log1p(values)
    c = z.corr(method=cfg["graph"]["corr"]["method"]).values
    c = np.nan_to_num(c, nan=0.0)
    if cfg["graph"]["corr"]["negative"] == "relu":
        c = np.maximum(c, 0.0)
    else:
        c = np.abs(c)
    c = sparsify_topk(c, cfg["graph"]["corr"]["top_k"])
    return normalize01(c)


def stats(a: np.ndarray) -> dict:
    off = a[~np.eye(len(a), dtype=bool)]
    nz = off > 0
    deg = (a > 0).sum(axis=1)
    return {
        "sparsity": float(1 - nz.mean()),
        "mean_degree": float(deg.mean()),
        "mean_weight_nz": float(off[nz].mean()) if nz.any() else 0.0,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(config: str) -> None:
    cfg = utils.load_config(config)
    utils.set_seed(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])
    fig_dir = Path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_parquet(proc / "traffic_matrix.parquet")
    node_index = utils.load_json(proc / "node_index.json")
    dates = pd.to_datetime(utils.load_json(proc / "dates.json")["dates"])
    nodes = list(matrix.columns)
    lat = np.array([node_index[c]["lat"] for c in nodes])
    lon = np.array([node_index[c]["lon"] for c in nodes])

    # in-regime base train span: first train_ratio of the timeline
    n_train = int(len(matrix) * cfg["split"]["train_ratio"])
    train_slice = matrix.iloc[:n_train]
    print(f"[graphs] N={len(nodes)} | corr train span: "
          f"{dates[0].date()} → {dates[n_train-1].date()} ({n_train} days)")

    saved = {}

    # (a) geo --------------------------------------------------------------- #
    a_geo, sigma = geo_graph(lat, lon, cfg)
    saved["A_geo"] = a_geo
    print(f"[geo]  sigma={sigma:.0f} km | {stats(a_geo)}")

    # (b) corr — base train span ------------------------------------------- #
    a_corr = corr_graph(train_slice, cfg)
    saved["A_corr"] = a_corr
    print(f"[corr] base train | {stats(a_corr)}")

    # (c) mixed ------------------------------------------------------------- #
    alpha, beta = cfg["graph"]["mix"]["alpha"], cfg["graph"]["mix"]["beta"]
    a_mixed = normalize01(alpha * a_geo + beta * a_corr)
    saved["A_mixed"] = a_mixed
    print(f"[mix]  alpha={alpha} beta={beta} | {stats(a_mixed)}")

    # (d) regime-conditional corr graphs (C2/C3) --------------------------- #
    regime_arr = np.array(utils.load_json(proc / "dates.json")["regime"])
    for regime in cfg["regimes"]:
        sel = regime_arr == regime
        if sel.sum() < 30:
            continue
        a_r = corr_graph(matrix.iloc[sel], cfg)
        saved[f"A_corr_{regime}"] = a_r
        saved[f"A_mixed_{regime}"] = normalize01(alpha * a_geo + beta * a_r)
        print(f"[corr:{regime}] {int(sel.sum())} days | {stats(a_r)}")

    # --- persist (+ normalized forms) + stats log ------------------------- #
    Path(cfg["paths"]["results"]).mkdir(parents=True, exist_ok=True)
    stat_rows = []
    for name, a in saved.items():
        np.save(proc / f"{name}.npy", a.astype(np.float32))
        np.save(proc / f"{name}_hat.npy", normalized_adjacency(a).astype(np.float32))
        stat_rows.append({"graph": name, **stats(a)})
    pd.DataFrame(stat_rows).to_csv(Path(cfg["paths"]["results"]) / "graph_stats.csv",
                                   index=False)

    # --- heatmaps --------------------------------------------------------- #
    _plot_heatmaps(saved, fig_dir)
    print(f"[done] saved {len(saved)} graphs (+ normalized) to {proc}")


def _plot_heatmaps(saved: dict, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["A_geo", "A_corr", "A_mixed"]
    keys = [k for k in keys if k in saved]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        im = ax.imshow(saved[k], cmap="viridis", aspect="auto")
        ax.set_title(k)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Latent airport dependency graphs")
    fig.tight_layout()
    fig.savefig(fig_dir / "graphs_heatmap.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
