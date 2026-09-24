"""The shift/amplitude diagnostic — computed BEFORE training, on purpose.

Everything is measured in the space the models actually see: log1p + per-node
z-score whose statistics are fit on the scenario's TRAIN rows only. For
``cross_pre2shock`` that train span is pre-COVID, so a unit here is one
pre-COVID standard deviation of that node.

  level(t)  cross-node mean of the scaled matrix on day t (the system level)
  shift     mean level over TEST days minus mean level over TRAIN days
  amp       peak-to-trough range of level(t) WITHIN the train span, i.e. how
            much level movement the model already saw while fitting
  shift/amp the controlling quantity: a shock only breaks a pre-learned mapping
            when it is large *relative to the level variation in training*

Usage:  python3 src/shift_amp.py configs/cta.yaml [configs/base.yaml ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import utils

SCEN = "cross_pre2shock"


def _window(proc, te, sreg):
    """Restrict the evaluation days to the regime the errors are measured on.

    Airports is the only dataset whose shock span runs two calendar years, and
    its 2021 half is largely back to normal. Diagnosing the full span while the
    errors come from 2020 alone reports a shift the performance numbers never
    met: 3.102 instead of 3.845. ``sreg`` names the window, e.g. "shock2020".
    """
    if sreg in (None, "all"):
        return te
    dj = utils.load_json(proc / "dates.json")
    reg = np.array(dj["regime"])
    year = sreg[-4:]
    if not year.isdigit():
        sel = [t for t in te if reg[t] == sreg]
        return np.asarray(sel, dtype=int) if sel else te
    yrs = pd.to_datetime(dj["dates"]).year
    sel = [t for t in te if reg[t] == sreg[:-4] and yrs[t] == int(year)]
    return np.asarray(sel, dtype=int) if sel else te


def diagnose(config: str, sreg: str = "all") -> dict:
    cfg = utils.load_config(config)
    proc = Path(cfg["paths"]["processed"])
    matrix = pd.read_parquet(proc / "traffic_matrix.parquet")
    sp = utils.load_json(proc / "splits.json")["scenarios"][SCEN]
    scaler = utils.TrafficScaler.from_dict(sp["scaler"])
    S = scaler.transform(matrix.values)                 # NaN preserved

    tr = np.asarray(sp["train"] + sp["val"])
    te = _window(proc, np.asarray(sp["test"]), sreg)
    level = np.nanmean(S, axis=1)                       # system level per day
    lt, ls = level[tr], level[te]
    shift = float(np.nanmean(ls) - np.nanmean(lt))
    amp = float(np.nanmax(lt) - np.nanmin(lt))

    lo, hi = np.nanpercentile(S[tr], [0.5, 99.5])
    cells = S[te]
    outside = float(np.nanmean((cells < lo) | (cells > hi)) * 100)
    below = float(np.mean(ls < np.nanmin(lt)) * 100)
    return {"dataset": Path(config).stem, "N": matrix.shape[1],
            "shift": shift, "amp": amp, "shift/amp": abs(shift) / amp,
            "cells_outside_p0.5_99.5_%": outside,
            "test_days_below_train_min_%": below}


if __name__ == "__main__":
    rows = [diagnose(c) for c in (sys.argv[1:] or ["configs/base.yaml"])]
    df = pd.DataFrame(rows).sort_values("shift/amp")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
