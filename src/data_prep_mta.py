"""Phase 1 (dataset #3) — NYC subway station activity matrix from turnstile audits.

Chosen by the selection criterion that came out of dataset #2: what breaks
pre-learned mappings is not shock size but **shift relative to the natural
in-training level amplitude**. NYC subway has a near-aseasonal baseline and a
−90% drop sustained for over a year, so its ratio is far above both airports
and bikeshare.

Raw: MTA Subway Turnstile Usage Data (data.ny.gov), one gzipped CSV per year in
``data/raw_mta/turnstile_YYYY.csv.gz`` with columns

    C/A, Unit, SCP, Station, Line Name, Division, Date, Time, Description,
    Entries, Exits

``Entries``/``Exits`` are **cumulative counters per turnstile**, audited roughly
every 4 hours, so the signal has to be recovered by differencing each
(C/A, Unit, SCP) series in time order. Counters run backwards on some units and
reset arbitrarily, hence the clamping below.

Node = station complex, keyed by (Station, Line Name, Division) — the same key
the MTA uses to group turnstiles. Signal = daily entries + exits, i.e. total
station activity, the analogue of ``FLT_TOT_1`` (arr+dep) and of the bikeshare
starts+ends. No OD information is used.

Coordinates come from the ``MTA Subway Stations`` reference (39hk-dx4f), matched
to turnstile station names by normalized-name similarity plus served-route
overlap; unmatched stations are dropped and reported.
"""
from __future__ import annotations

import argparse
import array
import collections
import csv
import datetime
import difflib
import glob
import gzip
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import utils

# A turnstile audit covers ~4 h; anything larger than this is a counter
# reset/rollover rather than real ridership.
MAX_DIFF_PER_AUDIT = 5000
# Consecutive audits for one turnstile are 4 hours apart (median 240 min, p99.9
# 480 min; only 0.011% of diffs exceed 24 h). A diff spanning longer than this
# is not a daily total — it is the whole missing stretch dumped onto the day the
# feed resumed. The archive has eight 7-day outages, and the magnitude filter
# alone only catches them while ridership is high: in 2020, with ridership at a
# quarter of normal, a 7-day accumulation slipped under MAX_DIFF_PER_AUDIT and
# inflated 2020-08-15 and 2020-10-03 by 4.8x and 4.0x — inside the shock test
# window. Guarding on elapsed time removes them at the source.
MAX_GAP_MIN = 1440

ABBREV = {
    "STREET": "ST", "AVENUE": "AV", "AVE": "AV", "SQUARE": "SQ",
    "ROAD": "RD", "PLACE": "PL", "PARKWAY": "PKWY", "BOULEVARD": "BLVD",
    "BLVD": "BLVD", "HEIGHTS": "HTS", "CENTER": "CTR", "CENTRE": "CTR",
    "TERMINAL": "TERM", "STATION": "STA", "BEACH": "BCH", "PARK": "PK",
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "FORT": "FT",
    "MOUNT": "MT", "SAINT": "ST",
}


def norm_name(s: str) -> list[str]:
    s = s.upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    # split letter/digit runs so "CENTRAL PK N110" meets "Central Park North (110 St)"
    s = re.sub(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", " ", s)
    toks = []
    for t in s.split():
        t = re.sub(r"(\d+)(ST|ND|RD|TH)$", r"\1", t)     # 14TH -> 14
        toks.append(ABBREV.get(t, t))
    return toks


def routes_of(line_name: str) -> set[str]:
    return set(re.findall(r"[A-Z0-9]", (line_name or "").upper()))


# --------------------------------------------------------------------------- #
# Turnstile differencing
# --------------------------------------------------------------------------- #
def _minute_of_day(tm: str) -> int:
    """Minutes since midnight from an MTA turnstile ``Time`` field.

    The feed is 24-hour (``"16:00:00"``) for almost the whole archive, but for
    2021-07-24..2021-08-06 it switches to 12-hour with a suffix
    (``"4:00:00 PM"``). Reading ``int(hh)`` alone maps 4 PM and 4 AM — and
    midnight and noon — onto the same minute, which collapses each turnstile's
    six daily audits into three tied timestamps. The tie makes the sort fall
    back on file order, so the ``abs`` differencing below zig-zags up and down
    the counter and roughly triples that fortnight's activity.
    """
    tm = tm.strip()
    suffix = tm[-2:].upper()
    if suffix in ("AM", "PM"):
        hh, mm, _ = tm[:-2].strip().split(":")
        hh = int(hh) % 12                      # 12 AM -> 0, 12 PM -> 12 (below)
        if suffix == "PM":
            hh += 12
    else:
        hh, mm, _ = tm.split(":")
        hh = int(hh)
    return hh * 60 + int(mm)


def read_year(path: str, station_ids: dict, day_ids: dict):
    """Stream one yearly CSV and return per-(station, day) total activity.

    Everything is accumulated into flat integer arrays first, then sorted by
    (turnstile, timestamp) so consecutive cumulative readings can be differenced
    inside each turnstile's own series.
    """
    ts_ids: dict = {}
    ts_station = array.array("i")           # turnstile -> station id
    t_id = array.array("i")                 # per row
    t_min = array.array("q")                # true calendar minutes (must be
                                            # chronological: the diffing below
                                            # relies on this ordering)
    t_ent = array.array("q")
    t_ext = array.array("q")
    t_day = array.array("i")
    day_cache: dict = {}                    # "MM/DD/YYYY" -> (day id, ordinal)
    tod_cache: dict = {}                    # time string -> minute of day

    n = 0
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        col = {h: i for i, h in enumerate(header)}
        i_ca, i_un, i_scp = col["C/A"], col["Unit"], col["SCP"]
        i_st, i_ln, i_dv = col["Station"], col["Line Name"], col["Division"]
        i_dt, i_tm = col["Date"], col["Time"]
        i_en, i_ex = col["Entries"], col["Exits"]
        for row in reader:
            try:
                ent, ext = int(row[i_en]), int(row[i_ex])
            except (ValueError, IndexError):
                continue
            skey = (row[i_st].strip(), row[i_ln].strip(), row[i_dv].strip())
            sid = station_ids.setdefault(skey, len(station_ids))
            tkey = (row[i_ca], row[i_un], row[i_scp])
            tid = ts_ids.get(tkey)
            if tid is None:
                tid = ts_ids[tkey] = len(ts_ids)
                ts_station.append(sid)
            cached = day_cache.get(row[i_dt])
            if cached is None:
                m, d, y = row[i_dt].split("/")
                did = day_ids.setdefault(f"{y}-{m}-{d}", len(day_ids))
                ordn = datetime.date(int(y), int(m), int(d)).toordinal()
                cached = day_cache[row[i_dt]] = (did, ordn)
            did, ordn = cached
            tod = tod_cache.get(row[i_tm])
            if tod is None:
                tod = tod_cache[row[i_tm]] = _minute_of_day(row[i_tm])
            t_id.append(tid)
            t_min.append(ordn * 1440 + tod)
            t_ent.append(ent)
            t_ext.append(ext)
            t_day.append(did)
            n += 1

    tid_a = np.frombuffer(t_id, dtype=np.int32)
    min_a = np.frombuffer(t_min, dtype=np.int64)
    ent_a = np.frombuffer(t_ent, dtype=np.int64)
    ext_a = np.frombuffer(t_ext, dtype=np.int64)
    day_a = np.frombuffer(t_day, dtype=np.int32)

    order = np.lexsort((min_a, tid_a))
    tid_s, ent_s, ext_s, day_s = tid_a[order], ent_a[order], ext_a[order], day_a[order]
    min_s = min_a[order]

    same = tid_s[1:] == tid_s[:-1]                    # consecutive same turnstile
    d_ent = np.abs(ent_s[1:] - ent_s[:-1])            # counters may run backwards
    d_ext = np.abs(ext_s[1:] - ext_s[:-1])
    d_min = min_s[1:] - min_s[:-1]                    # elapsed time between audits
    ok = (same & (d_ent <= MAX_DIFF_PER_AUDIT) & (d_ext <= MAX_DIFF_PER_AUDIT)
          & (d_min <= MAX_GAP_MIN))
    activity = (d_ent + d_ext)[ok]
    sids = np.asarray(ts_station, dtype=np.int32)[tid_s[1:][ok]]
    days = day_s[1:][ok]

    dropped = int((~ok).sum() - (~same).sum())
    print(f"  [read] {Path(path).name}: {n:,} rows, {len(ts_ids):,} turnstiles, "
          f"{dropped:,} anomalous diffs dropped "
          f"({dropped/max(len(same),1)*100:.1f}%)")
    return sids, days, activity


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #
def match_coords(station_keys: list[tuple], ref_path: str):
    """Fuzzy-match turnstile station keys to the GTFS station reference."""
    ref = json.load(open(ref_path))
    cands = []
    for r in ref:
        if "gtfs_latitude" not in r:
            continue
        cands.append({
            "toks": norm_name(r["stop_name"]),
            "routes": routes_of(r.get("daytime_routes", "").replace(" ", "")),
            "div": r.get("division", ""),
            "lat": float(r["gtfs_latitude"]),
            "lon": float(r["gtfs_longitude"]),
        })

    out, unmatched = {}, []
    for key in station_keys:
        name, line, div = key
        toks = norm_name(name)
        routes = routes_of(line)
        best, best_score = None, 0.0
        for c in cands:
            name_sim = difflib.SequenceMatcher(
                None, " ".join(toks), " ".join(c["toks"])).ratio()
            r_ov = (len(routes & c["routes"]) / len(c["routes"])
                    if c["routes"] else 0.0)
            score = 0.7 * name_sim + 0.25 * r_ov + 0.05 * (c["div"] == div)
            if score > best_score:
                best, best_score = c, score
        if best is not None and best_score >= 0.62:
            out[key] = (best["lat"], best["lon"], best_score)
        else:
            unmatched.append((key, round(best_score, 2)))
    return out, unmatched


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
        raise FileNotFoundError(f"no files matched {cfg['paths']['raw_glob']}")
    print(f"[load] {len(files)} yearly archives, span {lo} → {hi}")

    station_ids: dict = {}
    day_ids: dict = {}
    blocks = [read_year(f, station_ids, day_ids) for f in files]

    days = pd.date_range(lo, hi, freq="D")
    day_pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(days)}
    # map the file-order day ids onto the calendar grid (-1 = outside the span)
    id2pos = np.full(len(day_ids), -1, dtype=np.int64)
    for d, i in day_ids.items():
        id2pos[i] = day_pos.get(d, -1)

    m = np.zeros((len(days), len(station_ids)), dtype=np.float64)
    for sids, dids, act in blocks:
        pos = id2pos[dids]
        keep = pos >= 0
        np.add.at(m, (pos[keep], sids[keep]), act[keep])
    keys = [k for k, _ in sorted(station_ids.items(), key=lambda kv: kv[1])]
    print(f"[pivot] full grid: {m.shape[0]} days × {m.shape[1]} station keys")

    # days with no audit at all for a station are gaps, not zeros
    m[m == 0] = np.nan
    matrix = pd.DataFrame(m, index=days, columns=[f"{a}|{b}|{c}" for a, b, c in keys])

    coverage = matrix.notna().mean(axis=0)
    thr = cfg["data"]["coverage_threshold"]
    keep_cols = coverage[coverage >= thr].index
    print(f"[coverage] threshold={thr}: kept {len(keep_cols)}, "
          f"dropped {matrix.shape[1] - len(keep_cols)}")
    matrix = matrix[keep_cols]

    floor = cfg["data"]["min_median_activity"]
    med = matrix.median(axis=0)
    alive = med[med >= floor].index
    print(f"[activity] median >= {floor}: kept {len(alive)}, "
          f"dropped {matrix.shape[1] - len(alive)}")
    matrix = matrix[alive]

    # --- coordinates ------------------------------------------------------- #
    live_keys = [tuple(c.split("|")) for c in matrix.columns]
    coords, unmatched = match_coords(live_keys, cfg["paths"]["stations_ref"])
    print(f"[coords] matched {len(coords)}/{len(live_keys)} stations")
    if unmatched:
        print(f"[coords] dropped {len(unmatched)}: "
              f"{[u[0][0] for u in unmatched[:15]]}"
              + (" ..." if len(unmatched) > 15 else ""))
    matrix = matrix[[c for c in matrix.columns
                     if tuple(c.split("|")) in coords]]

    # --- node metadata ----------------------------------------------------- #
    hub_q = matrix.mean(axis=0).quantile(cfg["mask"]["hub_quantile"])
    node_index = {}
    for i, c in enumerate(matrix.columns):
        lat, lon, score = coords[tuple(c.split("|"))]
        node_index[c] = {
            "index": i, "name": c.split("|")[0], "state": "NY",
            "lat": lat, "lon": lon,
            "coverage": float(coverage[c]),
            "mean_traffic": float(matrix[c].mean()),
            "is_hub": bool(matrix[c].mean() >= hub_q),
            "match_score": round(score, 3),
        }
    print(f"[nodes] final N={matrix.shape[1]} "
          f"({sum(v['is_hub'] for v in node_index.values())} hubs, "
          f"hub_q={hub_q:.0f} riders/day)")

    dates = [d.strftime("%Y-%m-%d") for d in matrix.index]
    regime = [utils.regime_of_year(d.year, cfg["regimes"]) for d in matrix.index]
    print(f"[save] regimes: {pd.Series(regime).value_counts().to_dict()}")

    reg = np.array(regime)
    for r in ("pre", "shock", "recovery"):
        sel = reg == r
        if sel.any():
            print(f"[level] {r:8s} mean station-day activity = "
                  f"{np.nanmean(matrix.values[sel]):9.1f}")
    print(f"[level] 2020-04 (lockdown trough)      = "
          f"{np.nanmean(matrix.loc['2020-04-01':'2020-04-30'].values):9.1f}")

    matrix.columns.name = "station"
    matrix.to_parquet(proc / "traffic_matrix.parquet")
    utils.save_json(node_index, proc / "node_index.json")
    utils.save_json({"dates": dates, "regime": regime}, proc / "dates.json")
    print(f"[done] N={matrix.shape[1]}, T={matrix.shape[0]}, "
          f"NaN cells={int(matrix.isna().sum().sum()):,} "
          f"({matrix.isna().mean().mean()*100:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mta.yaml")
    main(ap.parse_args().config)
