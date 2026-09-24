"""Phase 1 (dataset #4) — US EPA daily NO2 monitor network.

Fourth dataset, chosen to leave the transport domain entirely: air-quality
sensing is the canonical spatio-temporal kriging benchmark family (AQI-36,
KnowAir), so the claim has to survive there. It is also a **pre-registered test
of the selection criterion** from dataset #2: NO2 has a strong weather-driven
annual cycle, so the criterion predicts a *low* shift/amplitude ratio and hence
**no collapse** — the same signature bikeshare showed, in a different domain.

Raw: EPA AQS pre-generated daily summaries, one zip per year, parameter 42602
(NO2), in ``data/raw_aqs/daily_42602_YYYY.zip``. Columns already carry
``Latitude``/``Longitude``, so no coordinate join is needed.

Node = monitoring site ``(State Code, County Code, Site Num)``; multiple monitors
(POC) at one site are averaged. Signal = daily mean NO2 in ppb.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import utils


def stream(files: list[str], lo: str, hi: str):
    """Return {(site, day): [values]} and {site: (lat, lon, name)}."""
    vals: dict = defaultdict(list)
    meta: dict = {}
    for path in files:
        with zipfile.ZipFile(path) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(name) as fh:
                r = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8-sig"))
                n = 0
                for row in r:
                    d = row["Date Local"]
                    if not (lo <= d <= hi):
                        continue
                    site = (f'{row["State Code"]}-{row["County Code"]}'
                            f'-{row["Site Num"]}')
                    try:
                        v = float(row["Arithmetic Mean"])
                    except ValueError:
                        continue
                    vals[(site, d)].append(max(v, 0.0))   # clip sub-LOD negatives
                    if site not in meta:
                        meta[site] = (float(row["Latitude"]),
                                      float(row["Longitude"]),
                                      row.get("Local Site Name") or row["County Name"])
                    n += 1
        print(f"  [read] {Path(path).name}: {n:,} site-days in span")
    return vals, meta


def main(config: str) -> None:
    cfg = utils.load_config(config)
    utils.set_seed(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])
    proc.mkdir(parents=True, exist_ok=True)

    lo, hi = cfg["data"]["date_start"], cfg["data"]["date_end"]
    files = sorted(glob.glob(cfg["paths"]["raw_glob"]))
    if not files:
        raise FileNotFoundError(f"no zips matched {cfg['paths']['raw_glob']}")
    print(f"[load] {len(files)} yearly archives, span {lo} → {hi}")

    vals, meta = stream(files, lo, hi)
    days = pd.date_range(lo, hi, freq="D")
    day_pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(days)}
    sites = sorted({s for s, _ in vals})
    st_pos = {s: j for j, s in enumerate(sites)}

    m = np.full((len(days), len(sites)), np.nan, dtype=np.float32)
    for (s, d), v in vals.items():
        m[day_pos[d], st_pos[s]] = float(np.mean(v))       # average over POCs
    matrix = pd.DataFrame(m, index=days, columns=sites)
    print(f"[pivot] full grid: {matrix.shape[0]} days × {matrix.shape[1]} sites")

    coverage = matrix.notna().mean(axis=0)
    thr = cfg["data"]["coverage_threshold"]
    keep = coverage[coverage >= thr].index
    print(f"[coverage] threshold={thr}: kept {len(keep)}, "
          f"dropped {matrix.shape[1] - len(keep)} sites")
    matrix = matrix[keep]

    floor = cfg["data"]["min_median_activity"]
    med = matrix.median(axis=0)
    alive = med[med >= floor].index
    print(f"[activity] median >= {floor} ppb: kept {len(alive)}, "
          f"dropped {matrix.shape[1] - len(alive)} sites")
    matrix = matrix[alive]

    hub_q = matrix.mean(axis=0).quantile(cfg["mask"]["hub_quantile"])
    node_index = {}
    for i, s in enumerate(matrix.columns):
        lat, lon, nm = meta[s]
        node_index[s] = {
            "index": i, "name": nm, "state": s.split("-")[0],
            "lat": lat, "lon": lon,
            "coverage": float(coverage[s]),
            "mean_traffic": float(matrix[s].mean()),
            "is_hub": bool(matrix[s].mean() >= hub_q),
        }
    print(f"[nodes] final N={matrix.shape[1]} "
          f"({sum(v['is_hub'] for v in node_index.values())} hubs, "
          f"hub_q={hub_q:.1f} ppb)")

    dates = [d.strftime("%Y-%m-%d") for d in matrix.index]
    regime = [utils.regime_of_year(d.year, cfg["regimes"]) for d in matrix.index]
    print(f"[save] regimes: {pd.Series(regime).value_counts().to_dict()}")

    reg = np.array(regime)
    for r in ("pre", "shock", "recovery"):
        sel = reg == r
        if sel.any():
            print(f"[level] {r:8s} mean site-day NO2 = "
                  f"{np.nanmean(matrix.values[sel]):6.2f} ppb")
    print(f"[level] 2020-04 (lockdown trough)  = "
          f"{np.nanmean(matrix.loc['2020-04-01':'2020-04-30'].values):6.2f} ppb")

    matrix.columns.name = "site"
    matrix.to_parquet(proc / "traffic_matrix.parquet")
    utils.save_json(node_index, proc / "node_index.json")
    utils.save_json({"dates": dates, "regime": regime}, proc / "dates.json")
    print(f"[done] N={matrix.shape[1]}, T={matrix.shape[0]}, "
          f"NaN cells={int(matrix.isna().sum().sum()):,} "
          f"({matrix.isna().mean().mean()*100:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aqs.yaml")
    main(ap.parse_args().config)
