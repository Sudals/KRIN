"""Phase 3 — Temporal splits, sliding windows, and spatial masking.

We define reconstruction *scenarios* (leakage-safe by construction):

  inregime            chronological train/val/test over the full 2016–2026 line
  cross_pre2shock     train+val on pre-COVID, test on the 2020–21 shock
  cross_pre2recovery  train+val on pre-COVID, test on 2022–26 recovery

For each scenario we store the target-day indices, the regime-appropriate
correlation graph to use, and a scaler fit on that scenario's TRAIN rows only.

Masking is the node-level "which airports are observed" pattern. It is shared
across scenarios and fixed per (ratio, strategy, seed): a chosen subset of
airports is hidden for the entire evaluation, mirroring inductive kriging where
some stations simply have no sensor. Reproducible from the seed alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import utils


def _valid_targets(idx: np.ndarray, K: int, all_len: int) -> list[int]:
    """Target days that have a full K-day history available."""
    return [int(t) for t in idx if t - K >= 0 and t < all_len]


def build_scenarios(matrix, regime_arr, cfg) -> dict:
    K = cfg["split"]["window_K"]
    T = len(matrix)
    scenarios: dict = {}

    # --- in-regime: chronological over the whole timeline ----------------- #
    n_train = int(T * cfg["split"]["train_ratio"])
    n_val = int(T * cfg["split"]["val_ratio"])
    scenarios["inregime"] = {
        "train_rows": np.arange(0, n_train),
        "train": _valid_targets(np.arange(K, n_train), K, T),
        "val": _valid_targets(np.arange(n_train, n_train + n_val), K, T),
        "test": _valid_targets(np.arange(n_train + n_val, T), K, T),
        "graph": "A_mixed",
    }

    # --- cross-regime: train on pre, test on a later regime --------------- #
    pre_idx = np.where(regime_arr == "pre")[0]
    n_pre_train = int(len(pre_idx) * 0.85)          # hold out tail of pre for val
    pre_train, pre_val = pre_idx[:n_pre_train], pre_idx[n_pre_train:]
    for test_regime in ("shock", "recovery"):
        test_idx = np.where(regime_arr == test_regime)[0]
        scenarios[f"cross_pre2{test_regime}"] = {
            "train_rows": pre_train,
            "train": _valid_targets(pre_train, K, T),
            "val": _valid_targets(pre_val, K, T),
            "test": _valid_targets(test_idx, K, T),
            "graph": "A_mixed_pre",                 # built from pre only → safe
        }
    return scenarios


def fit_scaler(matrix, train_rows, cfg) -> utils.TrafficScaler:
    return utils.TrafficScaler(
        log1p=cfg["transform"]["log1p"], zscore=cfg["transform"]["zscore"]
    ).fit(matrix.values[train_rows])


def build_masks(node_index, cfg) -> dict:
    nodes = sorted(node_index, key=lambda c: node_index[c]["index"])
    n = len(nodes)
    hub_flags = np.array([node_index[c]["is_hub"] for c in nodes])
    masks: dict = {}
    for r in cfg["mask"]["ratios"]:
        masks[str(r)] = {}
        for strat in cfg["mask"]["strategies"]:
            masks[str(r)][strat] = {}
            for seed in cfg["mask"]["seeds"]:
                obs = utils.spatial_mask(n, r, seed, strat, hub_flags)
                masks[str(r)][strat][str(seed)] = obs.astype(int).tolist()
    return masks


def main(config: str) -> None:
    cfg = utils.load_config(config)
    utils.set_seed(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])

    matrix = pd.read_parquet(proc / "traffic_matrix.parquet")
    node_index = utils.load_json(proc / "node_index.json")
    regime_arr = np.array(utils.load_json(proc / "dates.json")["regime"])
    K = cfg["split"]["window_K"]

    scenarios = build_scenarios(matrix, regime_arr, cfg)
    out = {"window_K": K, "scenarios": {}}
    for name, sc in scenarios.items():
        scaler = fit_scaler(matrix, sc["train_rows"], cfg)
        out["scenarios"][name] = {
            "train": sc["train"], "val": sc["val"], "test": sc["test"],
            "graph": sc["graph"], "scaler": scaler.to_dict(),
        }
        print(f"[{name}] train={len(sc['train'])} val={len(sc['val'])} "
              f"test={len(sc['test'])} | graph={sc['graph']}")

    masks = build_masks(node_index, cfg)
    # reproducibility check + hidden-count log
    n = len(node_index)
    for r in cfg["mask"]["ratios"]:
        for strat in cfg["mask"]["strategies"]:
            obs0 = np.array(masks[str(r)][strat]["0"])
            again = utils.spatial_mask(
                n, r, 0, strat,
                np.array([node_index[c]["is_hub"]
                          for c in sorted(node_index, key=lambda c: node_index[c]["index"])]))
            assert (obs0 == again.astype(int)).all(), "mask not reproducible!"
            print(f"[mask] r={r} {strat}: hidden={n - int(obs0.sum())}/{n} (seed0, reproducible ✓)")

    utils.save_json(out, proc / "splits.json")
    utils.save_json(masks, proc / "masks.json")
    print(f"[done] splits.json + masks.json written ({len(scenarios)} scenarios)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
