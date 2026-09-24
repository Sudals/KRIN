"""Phase 4 — Non-neural baselines (sanity-check lower bounds).

All operate directly in the original traffic scale and are evaluated on the
HIDDEN nodes only:

  HA            per-node historical average over the K-day window
  Naive-1     the hidden node's OWN value one day earlier
  SNaive-7     the hidden node's OWN value one week earlier
  KNN-geo       distance-weighted mean of OBSERVED neighbours on day t (A_geo)
  KNN-mixed     same, using the mixed graph A (geo + correlation)

Naive-k matters more than it looks. Only day *t* is hidden for a masked node —
its history stays visible to every model — so copying the node's own value from
k days back is available to any method and needs no learning at all. On the
shock split SNaive-7 beats most of the learned models, which is why it belongs
in the main table rather than an appendix.

These run across every (scenario × ratio × strategy × seed) and write
results/baselines.csv with an overall row plus per-regime breakdown.
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import utils
from kriging_data import load_bundle


def predict_ha(raw, targets, K, observed) -> np.ndarray:
    """(n_targets, N) predictions = node mean over its K-day history."""
    preds = np.empty((len(targets), raw.shape[1]))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for r, t in enumerate(targets):
            preds[r] = np.nanmean(raw[t - K:t], axis=0)
    return np.nan_to_num(preds, nan=0.0)


def predict_knn(raw, targets, weight, observed) -> np.ndarray:
    """Distance/affinity-weighted mean of observed neighbours on each day."""
    W = weight.copy()
    W[:, ~observed] = 0.0                       # only observed nodes contribute
    np.fill_diagonal(W, 0.0)
    preds = np.empty((len(targets), raw.shape[1]))
    for r, t in enumerate(targets):
        vals = np.nan_to_num(raw[t], nan=0.0)
        present = observed & ~np.isnan(raw[t])
        Wt = W * present[None, :]
        denom = Wt.sum(axis=1)
        num = Wt @ vals
        preds[r] = np.where(denom > 0, num / np.maximum(denom, 1e-9),
                            np.nanmean(vals[present]) if present.any() else 0.0)
    return preds


def evaluate_block(preds, raw, targets, hidden, regime_arr) -> list[dict]:
    """Overall + per-regime metrics on hidden nodes."""
    yt = raw[targets]                            # (n, N)
    hid = np.broadcast_to(hidden, yt.shape)
    rows = [{"regime": "all", **utils.metrics(yt, preds, hid)}]
    regs = regime_arr[targets]
    for reg in pd.unique(regs):
        sel = regs == reg
        rows.append({"regime": reg,
                     **utils.metrics(yt[sel], preds[sel], hid[sel])})
    return rows


def predict_persist(raw, targets, lag: int) -> np.ndarray:
    """(n_targets, N) predictions = each node's own value ``lag`` days earlier.

    Uses only the node's own history, which is visible even for hidden nodes, so
    it is a legitimate zero-parameter competitor rather than an oracle.
    """
    idx = np.asarray(targets) - lag
    return raw[np.clip(idx, 0, len(raw) - 1)]


def predict_common_mode(raw, targets, observed, lag: int = 7) -> np.ndarray:
    """SNaive-k corrected by the same-day common-mode change of the observed nodes.

    log(1+x_i_t) = log(1+x_i_{t-lag}) + median over observed j of
                   [ log(1+x_j_t) - log(1+x_j_{t-lag}) ]

    Still training-free and graph-free: it reads the node's own lagged value,
    which every method already gets, plus the observed nodes on day t, which
    every method already gets as input. It is the honest opponent for a paper
    about level shift, because unlike plain persistence it tracks the level.
    """
    L = np.log1p(np.clip(raw, 0, None))
    preds = np.empty((len(targets), raw.shape[1]))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for r, t in enumerate(targets):
            back = max(t - lag, 0)
            d = L[t] - L[back]
            d = np.where(observed, d, np.nan)
            shift = np.nanmedian(d) if np.isfinite(d).any() else 0.0
            preds[r] = np.expm1(L[back] + shift)
    # NaN where the lagged value is missing, exactly as predict_persist leaves it.
    # utils.metrics drops NaN predictions, so filling them with 0 would score a
    # fabricated zero against real traffic and inflate the error on the datasets
    # with internal gaps.
    return np.clip(preds, 0, None)


def main(config: str, split_shock_years: bool = False, mask_seeds=None) -> None:
    cfg = utils.load_config(config)
    proc = Path(cfg["paths"]["processed"])
    res = Path(cfg["paths"]["results"]); res.mkdir(parents=True, exist_ok=True)
    regime_arr = np.array(utils.load_json(proc / "dates.json")["regime"])
    if split_shock_years:
        # Must mirror run_main.py's --split-shock-years: if the models report a
        # 2020-only slice on airports, Naive-k and HA have to be available on the
        # SAME slice or the "only model that beats SNaive-7" claim cannot be made
        # on it.
        yrs = pd.to_datetime(utils.load_json(proc / "dates.json")["dates"]).year
        regime_arr = np.array([f"{r}{y}" if r == "shock" else r
                               for r, y in zip(regime_arr, yrs)])

    splits = utils.load_json(proc / "splits.json")["scenarios"]
    scenarios = list(splits)
    A_geo = np.load(proc / "A_geo.npy")
    records = []
    for scenario, ratio, strat, seed in itertools.product(
            scenarios, cfg["mask"]["ratios"], cfg["mask"]["strategies"],
            mask_seeds or cfg["mask"]["seeds"]):
        b = load_bundle(cfg, scenario, ratio, strat, seed)
        A_mixed = np.load(proc / f"{splits[scenario]['graph']}.npy")
        targets = b.test
        models = {
            "HA": predict_ha(b.raw, targets, b.K, b.observed),
            "Naive-1": predict_persist(b.raw, targets, 1),
            "SNaive-7": predict_persist(b.raw, targets, 7),
            "SNaive-7+CM": predict_common_mode(b.raw, targets, b.observed, 7),
            "KNN-geo": predict_knn(b.raw, targets, A_geo, b.observed),
            "KNN-mixed": predict_knn(b.raw, targets, A_mixed, b.observed),
        }
        for model, preds in models.items():
            for row in evaluate_block(preds, b.raw, targets, b.hidden, regime_arr):
                records.append({"model": model, "scenario": scenario,
                                "ratio": ratio, "strategy": strat, "seed": seed,
                                **row})
        print(f"[{scenario} r={ratio} {strat} s{seed}] "
              + " | ".join(f"{m}={utils.metrics(b.raw[targets], p, np.broadcast_to(b.hidden, (len(targets), len(b.nodes))))['MAE']:.1f}"
                           for m, p in models.items()))

    df = pd.DataFrame(records)
    df.to_csv(res / "baselines.csv", index=False)
    # tidy summary: mean over seeds, overall regime, random strategy
    summary = (df[(df.regime == "all") & (df.strategy == "random")]
               .groupby(["scenario", "model", "ratio"])[["MAE", "RMSE", "sMAPE", "R2"]]
               .mean().round(2))
    print("\n=== baseline summary (regime=all, random mask, mean over seeds) ===")
    print(summary.to_string())
    print(f"\n[done] {len(df)} rows → {res/'baselines.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split-shock-years", action="store_true",
                    help="partition the shock regime by calendar year (airports)")
    ap.add_argument("--mask-seeds", default=None,
                    help="e.g. 0-9 or 0,1,2 (default: config)")
    a = ap.parse_args()
    ms = None
    if a.mask_seeds:
        v = a.mask_seeds
        ms = ([int(x) for x in v.split(",")] if "," in v else
              list(range(int(v.split("-")[0]), int(v.split("-")[1]) + 1))
              if "-" in v else [int(v)])
    main(a.config, a.split_shock_years, ms)
