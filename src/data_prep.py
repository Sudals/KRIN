"""Phase 1 — Preprocessing & traffic matrix.

Loads the raw Eurocontrol CSVs (2016–2026), joins OurAirports coordinates,
filters nodes by day-coverage, and writes a clean (T × N) traffic matrix plus
a node index. No imputation and no scaling happen here — intra-series gaps are
recorded as NaN and scaling is fit later on the train span only.

Outputs (under ``data/processed``):
    traffic_matrix.parquet   rows=dates (DatetimeIndex), cols=ICAO, values=FLT_TOT_1
    node_index.json          ICAO ↔ index + name/state/lat/lon/hub/coverage
    dates.json               ordered date strings + per-date regime label
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import utils


def build_coords(cfg: dict) -> pd.DataFrame:
    """Extract ICAO → (lat, lon) from the OurAirports dump and cache it."""
    raw = pd.read_csv(cfg["paths"]["coords_raw"], low_memory=False)
    # `ident` is the ICAO-style identifier for the vast majority of rows.
    coords = (
        raw.rename(columns={"ident": "APT_ICAO",
                            "latitude_deg": "latitude",
                            "longitude_deg": "longitude"})
        [["APT_ICAO", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("APT_ICAO")
    )
    Path(cfg["paths"]["coords"]).parent.mkdir(parents=True, exist_ok=True)
    coords.to_csv(cfg["paths"]["coords"], index=False)
    return coords


def load_raw(cfg: dict) -> pd.DataFrame:
    files = sorted(glob.glob(cfg["paths"]["raw_glob"]))
    if not files:
        raise FileNotFoundError(f"no CSVs matched {cfg['paths']['raw_glob']}")
    usecols = [cfg["data"]["date_col"], cfg["data"]["node_key"],
               cfg["data"]["target_col"], "APT_NAME", "STATE_NAME"]
    frames = [pd.read_csv(f, usecols=usecols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df[cfg["data"]["date_col"]] = pd.to_datetime(df[cfg["data"]["date_col"]])
    print(f"[load] {len(files)} files, {len(df):,} rows, "
          f"{df[cfg['data']['node_key']].nunique()} airports, "
          f"{df[cfg['data']['date_col']].dt.year.min()}–{df[cfg['data']['date_col']].dt.year.max()}")
    return df


def main(config: str) -> None:
    cfg = utils.load_config(config)
    utils.set_seed(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])
    proc.mkdir(parents=True, exist_ok=True)

    date_col, node_key, target = (cfg["data"]["date_col"],
                                  cfg["data"]["node_key"], cfg["data"]["target_col"])

    df = load_raw(cfg)
    coords = build_coords(cfg)
    print(f"[coords] {len(coords):,} airports with lat/lon cached")

    # --- pivot to T × N (dates × airports) -------------------------------- #
    matrix = df.pivot_table(index=date_col, columns=node_key,
                            values=target, aggfunc="sum")
    matrix = matrix.sort_index()
    all_dates = pd.date_range(matrix.index.min(), matrix.index.max(), freq="D")
    matrix = matrix.reindex(all_dates)          # expose calendar gaps as NaN
    n_days = len(matrix)
    print(f"[pivot] full grid: {n_days} days × {matrix.shape[1]} airports")

    # --- coverage filtering ----------------------------------------------- #
    coverage = matrix.notna().mean(axis=0)
    thr = cfg["data"]["coverage_threshold"]
    keep = coverage[coverage >= thr].index
    dropped_cov = matrix.shape[1] - len(keep)
    matrix = matrix[keep]
    print(f"[coverage] threshold={thr}: kept {len(keep)}, dropped {dropped_cov} airports")

    # --- coordinate join -------------------------------------------------- #
    coord_map = coords.set_index("APT_ICAO")
    have_coords = [c for c in matrix.columns if c in coord_map.index]
    missing = sorted(set(matrix.columns) - set(have_coords))
    if missing:
        print(f"[coords] {len(missing)} nodes without coords, dropped: {missing[:20]}"
              + (" ..." if len(missing) > 20 else ""))
    matrix = matrix[have_coords]

    # --- node metadata ---------------------------------------------------- #
    meta = (df[[node_key, "APT_NAME", "STATE_NAME"]]
            .drop_duplicates(node_key).set_index(node_key))
    hub_q = matrix.mean(axis=0).quantile(cfg["mask"]["hub_quantile"])
    node_index = {}
    for i, icao in enumerate(matrix.columns):
        node_index[icao] = {
            "index": i,
            "name": str(meta.loc[icao, "APT_NAME"]) if icao in meta.index else "",
            "state": str(meta.loc[icao, "STATE_NAME"]) if icao in meta.index else "",
            "lat": float(coord_map.loc[icao, "latitude"]),
            "lon": float(coord_map.loc[icao, "longitude"]),
            "coverage": float(coverage[icao]),
            "mean_traffic": float(matrix[icao].mean()),
            "is_hub": bool(matrix[icao].mean() >= hub_q),
        }
    n_hub = sum(v["is_hub"] for v in node_index.values())
    print(f"[nodes] final N={matrix.shape[1]} ({n_hub} hubs, hub_q={hub_q:.0f} flights)")

    # --- per-date regime labels ------------------------------------------- #
    dates = [d.strftime("%Y-%m-%d") for d in matrix.index]
    regime = [utils.regime_of_year(d.year, cfg["regimes"]) for d in matrix.index]
    reg_counts = pd.Series(regime).value_counts().to_dict()

    # --- persist ---------------------------------------------------------- #
    matrix.columns.name = node_key
    matrix.to_parquet(proc / "traffic_matrix.parquet")
    utils.save_json(node_index, proc / "node_index.json")
    utils.save_json({"dates": dates, "regime": regime}, proc / "dates.json")

    print(f"[save] traffic_matrix.parquet shape={matrix.shape}")
    print(f"[save] regimes: {reg_counts}")
    print(f"[done] N={matrix.shape[1]}, T={matrix.shape[0]}, "
          f"NaN cells remaining={int(matrix.isna().sum().sum()):,} "
          f"({matrix.isna().mean().mean()*100:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
