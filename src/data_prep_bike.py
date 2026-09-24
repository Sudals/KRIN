"""Phase 1 (dataset #2) — Capital Bikeshare station activity matrix.

Second dataset for the regime-robustness claim: Washington DC bike-share.
Each *station* is a node; the signal is daily total activity (trips started +
trips ended at the station), the direct analogue of ``FLT_TOT_1`` (arrivals +
departures) at an airport. No OD information is used, matching the OD-free
setting of the airport experiments.

Raw files: ``data/raw_bike/YYYYMM-capitalbikeshare-tripdata.zip`` (one CSV each).
Two schemas exist and are both handled:

  2018-01 .. 2020-03   Duration, Start date, End date, Start station number, ...
  2020-04 ..           ride_id, started_at, ended_at, start_station_id,
                       start_lat, start_lng, ...

Station numbers (31xxx) are stable across the schema change, so the panel is
continuous; coordinates are taken from the newer files (median per station).

Writes the same three artifacts the airport pipeline consumes, so every
downstream stage (``build_graphs`` → ``splits`` → ``baselines`` → ``train`` →
``evaluate``) runs unchanged against ``configs/bike.yaml``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import utils

STATION_RE = re.compile(r"^\d+$")


# --------------------------------------------------------------------------- #
# Raw trip streaming
# --------------------------------------------------------------------------- #
def _fields(header: list[str]):
    """Return (start_id, start_date, end_id, end_date, lat, lon) column names."""
    if "Start station number" in header:
        return ("Start station number", "Start date",
                "End station number", "End date", None, None)
    return ("start_station_id", "started_at",
            "end_station_id", "ended_at", "start_lat", "start_lng")


def stream_counts(files: list[str], lo: str, hi: str):
    """One pass over every trip file.

    Returns
        counts  {(station, 'YYYY-MM-DD'): activity}   starts + ends
        coords  {station: [(lat, lon), ...]}          from the newer schema only
    """
    counts: collections.Counter = collections.Counter()
    coords: dict[str, list] = collections.defaultdict(list)

    for path in files:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist()
                     if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            n_rows = 0
            for name in names:
                with zf.open(name) as fh:
                    reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8-sig"))
                    header = next(reader)
                    sid, sdt, eid, edt, lat_c, lon_c = _fields(header)
                    col = {c: i for i, c in enumerate(header)}
                    i_sid, i_sdt = col[sid], col[sdt]
                    i_eid, i_edt = col[eid], col[edt]
                    i_lat = col.get(lat_c) if lat_c else None
                    i_lon = col.get(lon_c) if lon_c else None
                    for row in reader:
                        n_rows += 1
                        # trip start contributes to the start station's day
                        s, d = row[i_sid], row[i_sdt][:10]
                        if STATION_RE.match(s) and lo <= d <= hi:
                            counts[(s, d)] += 1
                            if i_lat is not None and len(coords[s]) < 50:
                                try:
                                    coords[s].append((float(row[i_lat]),
                                                      float(row[i_lon])))
                                except (ValueError, IndexError):
                                    pass
                        # trip end contributes to the end station's day
                        e, d2 = row[i_eid], row[i_edt][:10]
                        if STATION_RE.match(e) and lo <= d2 <= hi:
                            counts[(e, d2)] += 1
        print(f"  [read] {Path(path).name}: {n_rows:,} trips "
              f"(cum. {len(counts):,} station-days)")
    return counts, coords


def to_matrix(counts, lo: str, hi: str) -> pd.DataFrame:
    """Pivot to a (T × N) frame. A day inside a station's active span with no
    trips is a true zero; days outside the span are NaN (station not installed),
    mirroring the airport matrix where gaps stay NaN."""
    days = pd.date_range(lo, hi, freq="D")
    day_pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(days)}
    stations = sorted({s for s, _ in counts})
    st_pos = {s: j for j, s in enumerate(stations)}

    m = np.zeros((len(days), len(stations)), dtype=np.float32)
    for (s, d), c in counts.items():
        m[day_pos[d], st_pos[s]] = c

    # active span per station = first .. last day with any recorded trip
    seen = m > 0
    for j in range(m.shape[1]):
        idx = np.flatnonzero(seen[:, j])
        if len(idx) == 0:
            m[:, j] = np.nan
            continue
        m[: idx[0], j] = np.nan
        m[idx[-1] + 1:, j] = np.nan
    return pd.DataFrame(m, index=days, columns=stations)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(config: str) -> None:
    cfg = utils.load_config(config)
    utils.set_seed(cfg["seed"])
    proc = Path(cfg["paths"]["processed"])
    proc.mkdir(parents=True, exist_ok=True)

    lo, hi = cfg["data"]["date_start"], cfg["data"]["date_end"]
    files = sorted(glob.glob(cfg["paths"]["raw_glob"]))
    if not files:
        raise FileNotFoundError(f"no zips matched {cfg['paths']['raw_glob']}")
    print(f"[load] {len(files)} monthly archives, span {lo} → {hi}")

    counts, coords_raw = stream_counts(files, lo, hi)
    matrix = to_matrix(counts, lo, hi)
    print(f"[pivot] full grid: {matrix.shape[0]} days × {matrix.shape[1]} stations")

    # --- coverage filtering (same rule as the airport pipeline) ------------ #
    coverage = matrix.notna().mean(axis=0)
    thr = cfg["data"]["coverage_threshold"]
    keep = coverage[coverage >= thr].index
    print(f"[coverage] threshold={thr}: kept {len(keep)}, "
          f"dropped {matrix.shape[1] - len(keep)} stations")
    matrix = matrix[keep]

    # --- drop near-dead stations (median activity below a floor) ---------- #
    floor = cfg["data"]["min_median_activity"]
    med = matrix.median(axis=0)
    alive = med[med >= floor].index
    print(f"[activity] median >= {floor}: kept {len(alive)}, "
          f"dropped {matrix.shape[1] - len(alive)} stations")
    matrix = matrix[alive]

    # --- coordinates ------------------------------------------------------- #
    coord_map = {s: (float(np.median([p[0] for p in v])),
                     float(np.median([p[1] for p in v])))
                 for s, v in coords_raw.items() if v}
    have = [c for c in matrix.columns if c in coord_map]
    missing = sorted(set(matrix.columns) - set(have))
    if missing:
        print(f"[coords] {len(missing)} stations without coords, dropped: "
              f"{missing[:20]}" + (" ..." if len(missing) > 20 else ""))
    matrix = matrix[have]

    # --- node metadata ----------------------------------------------------- #
    hub_q = matrix.mean(axis=0).quantile(cfg["mask"]["hub_quantile"])
    node_index = {}
    for i, s in enumerate(matrix.columns):
        lat, lon = coord_map[s]
        node_index[s] = {
            "index": i,
            "name": f"station_{s}",
            "state": "DC",
            "lat": lat,
            "lon": lon,
            "coverage": float(coverage[s]),
            "mean_traffic": float(matrix[s].mean()),
            "is_hub": bool(matrix[s].mean() >= hub_q),
        }
    n_hub = sum(v["is_hub"] for v in node_index.values())
    print(f"[nodes] final N={matrix.shape[1]} ({n_hub} hubs, "
          f"hub_q={hub_q:.1f} trips/day)")

    # --- regimes ----------------------------------------------------------- #
    dates = [d.strftime("%Y-%m-%d") for d in matrix.index]
    regime = [utils.regime_of_year(d.year, cfg["regimes"]) for d in matrix.index]
    print(f"[save] regimes: {pd.Series(regime).value_counts().to_dict()}")

    # --- regime level check (the whole point of this dataset) -------------- #
    reg = np.array(regime)
    for r in ("pre", "shock", "recovery"):
        sel = reg == r
        if sel.any():
            print(f"[level] {r:8s} mean station-day activity = "
                  f"{np.nanmean(matrix.values[sel]):7.2f}")
    apr = matrix.loc["2020-04-01":"2020-04-30"]
    print(f"[level] 2020-04 (lockdown trough)      = {np.nanmean(apr.values):7.2f}")

    matrix.columns.name = "station_id"
    matrix.to_parquet(proc / "traffic_matrix.parquet")
    utils.save_json(node_index, proc / "node_index.json")
    utils.save_json({"dates": dates, "regime": regime}, proc / "dates.json")
    print(f"[done] N={matrix.shape[1]}, T={matrix.shape[0]}, "
          f"NaN cells={int(matrix.isna().sum().sum()):,} "
          f"({matrix.isna().mean().mean()*100:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bike.yaml")
    main(ap.parse_args().config)
