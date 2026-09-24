"""Phase 1 (dataset #4) — Chicago CTA 'L' station entries matrix.

Replaces the EPA NO2 dataset, which the regime audit showed carries essentially
no COVID shock (8 of its 12 labelled shock months sit at >=90% of their seasonal
norm). Chicago rapid transit does: entries fall ~80% in April 2020.

Each *station* (parent `map_id`, so both platform directions are one node) is a
node; the signal is daily total entries — a per-node total with no OD content,
matching `FLT_TOT_1` for airports and started+ended trips for bike-share.

Raw files, both public and unauthenticated (data.cityofchicago.org):
  data/raw_cta/cta_l_daily_2018_2021.csv   resource 5neh-572f, one row per
                                           (station_id, date) with `rides`
  data/raw_cta/cta_stations.csv            resource 8pix-ypme, stop-level, with
                                           `map_id` and a `location` POINT

Writes the same three artifacts as every other prep script, so build_graphs →
splits, baselines and run_main run unchanged against configs/cta.yaml.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import utils

POINT_RE = re.compile(r"\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)?")


def load_coords(path: str) -> dict:
    """map_id -> (lat, lon), median over the station's stops."""
    st = pd.read_csv(path, dtype=str)
    acc: dict = {}
    for mid, loc in zip(st["map_id"], st["location"]):
        if not isinstance(loc, str):
            continue
        m = POINT_RE.search(loc)
        if not m:
            continue
        acc.setdefault(str(mid), []).append((float(m.group(1)), float(m.group(2))))
    return {k: (float(np.median([p[0] for p in v])),
                float(np.median([p[1] for p in v]))) for k, v in acc.items()}


def to_matrix(path: str, lo: str, hi: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_id": str})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["rides"] = pd.to_numeric(df["rides"], errors="coerce")
    df = df[(df.date >= lo) & (df.date <= hi)]
    mat = df.pivot_table(index="date", columns="station_id", values="rides",
                         aggfunc="sum")
    # reindex onto the full calendar so genuinely missing days stay NaN
    full = pd.date_range(lo, hi, freq="D")
    return mat.reindex(full)


def main(config: str) -> None:
    cfg = utils.load_config(config)
    utils.set_seed(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])
    proc.mkdir(parents=True, exist_ok=True)

    lo, hi = cfg["data"]["date_start"], cfg["data"]["date_end"]
    print(f"[load] {cfg['paths']['rides']} span {lo} → {hi}")
    matrix = to_matrix(cfg["paths"]["rides"], lo, hi)
    print(f"[pivot] full grid: {matrix.shape[0]} days × {matrix.shape[1]} stations")

    coverage = matrix.notna().mean(axis=0)
    thr = cfg["data"]["coverage_threshold"]
    keep = coverage[coverage >= thr].index
    print(f"[coverage] threshold={thr}: kept {len(keep)}, "
          f"dropped {matrix.shape[1] - len(keep)}")
    matrix = matrix[keep]

    floor = cfg["data"]["min_median_activity"]
    med = matrix.median(axis=0)
    alive = med[med >= floor].index
    print(f"[activity] median >= {floor}: kept {len(alive)}, "
          f"dropped {matrix.shape[1] - len(alive)}")
    matrix = matrix[alive]

    coord_map = load_coords(cfg["paths"]["stations_ref"])
    have = [c for c in matrix.columns if c in coord_map]
    missing = sorted(set(matrix.columns) - set(have))
    print(f"[coords] matched {len(have)}/{matrix.shape[1]}"
          + (f", dropped {missing}" if missing else ""))
    matrix = matrix[have]

    hub_q = matrix.mean(axis=0).quantile(cfg["mask"]["hub_quantile"])
    node_index = {}
    for i, s in enumerate(matrix.columns):
        lat, lon = coord_map[s]
        node_index[s] = {
            "index": i, "name": f"cta_{s}", "state": "IL",
            "lat": lat, "lon": lon,
            "coverage": float(coverage[s]),
            "mean_traffic": float(matrix[s].mean()),
            "is_hub": bool(matrix[s].mean() >= hub_q),
        }
    n_hub = sum(v["is_hub"] for v in node_index.values())
    print(f"[nodes] final N={matrix.shape[1]} ({n_hub} hubs, "
          f"hub_q={hub_q:.0f} entries/day)")

    dates = [d.strftime("%Y-%m-%d") for d in matrix.index]
    regime = [utils.regime_of_year(d.year, cfg["regimes"]) for d in matrix.index]
    print(f"[save] regimes: {pd.Series(regime).value_counts().to_dict()}")

    reg = np.array(regime)
    for r in ("pre", "shock", "recovery"):
        sel = reg == r
        if sel.any():
            print(f"[level] {r:8s} mean station-day entries = "
                  f"{np.nanmean(matrix.values[sel]):8.1f}")
    apr = matrix.loc["2020-04-01":"2020-04-30"]
    print(f"[level] 2020-04 (lockdown trough)     = {np.nanmean(apr.values):8.1f}")

    matrix.columns.name = "station_id"
    matrix.to_parquet(proc / "traffic_matrix.parquet")
    utils.save_json(node_index, proc / "node_index.json")
    utils.save_json({"dates": dates, "regime": regime}, proc / "dates.json")
    print(f"[done] N={matrix.shape[1]}, T={matrix.shape[0]}, "
          f"NaN cells={int(matrix.isna().sum().sum()):,} "
          f"({matrix.isna().mean().mean()*100:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cta.yaml")
    main(ap.parse_args().config)
