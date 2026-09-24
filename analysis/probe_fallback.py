"""Item 1b: how often a hidden node has no observed history left in its window.

Under the block-hole protocol the node's own K-day window can be entirely holed,
and then the per-node level KRIN needs has nothing to average. This counts how
often that happens and reports what the code falls back to.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import utils
import kriging_data as kd
from kriging_data import load_bundle

CFG = [("bike", "Bikeshare"), ("cta", "Chicago"),
       ("base", "Airports"), ("mta", "Subway")]


def rates(cfg_path, pattern, rate):
    cfg = utils.load_config(cfg_path)
    kd.set_hist_missing(rate, pattern, 0, 7)
    b = load_bundle(cfg, "cross_pre2shock", 0.5, "random", 0, graph_override="A_geo")
    K = b.K
    obs = kd._hist_obs_mask(*b.raw.shape) if rate > 0 else None
    te = np.asarray(b.test)
    hid = np.flatnonzero(b.observed == 0)
    valid = ~np.isnan(b.raw)
    seen = valid if obs is None else (valid & obs)
    empty = partial = 0
    tot = 0
    for t in te:
        w = seen[t - K:t][:, hid]                      # (K, |hidden|)
        c = w.sum(0)
        tot += len(hid)
        empty += int((c == 0).sum())
        partial += int((c < K / 2).sum())
    return 100 * empty / tot, 100 * partial / tot


if __name__ == "__main__":
    print(f"{'':<11}{'protocol':<14}{'window fully empty':>22}"
          f"{'fewer than K/2 observed':>26}")
    for c, n in CFG:
        for pat, r, lab in (("point", 0.0, "main"), ("point", 0.25, "point 25%"),
                            ("block", 0.25, "block 25%")):
            e, p = rates(f"configs/{c}.yaml", pat, r)
            print(f"{n:<11}{lab:<12}{e:>18.2f}%{p:>18.2f}%")
