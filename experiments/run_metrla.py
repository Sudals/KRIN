"""Training-free baselines on METR-LA under the point-missing protocol.

The paper's claim is about protocol B, the setting GRIN and SPIN evaluate in:
missing entries are scattered over (sensor, step) pairs, so a masked sensor's
other steps stay observed and a persistence prediction is available for free.
Everything else in this paper is measured on our own datasets, which leaves the
claim one step removed from the benchmark it is about. This script closes that
step. It needs no training, so it can be run on the original data directly.

Protocol, following GRIN (Cini et al., ICLR 2022):
  * METR-LA, 34,272 five-minute steps x 207 sensors, zeros denote missing
  * point missing removes 25% of the AVAILABLE entries at random as targets
  * sequential 70/10/20 split; metrics on the test span only
  * MAE over the masked-and-available entries

Reported numbers for the trained methods are transcribed from the GRIN paper's
own table (arXiv:2108.00298, tab_irish_traffic.tex), not reproduced here. Two
checks confirm the protocol matches theirs: the original missing rate comes out
at 8.1% against their reported 8.10%, the injected faults at 23.0% against their
23.00%, and their training-free MEAN recomputed here is 7.51 against the 7.56
they report.

Usage:  python3 experiments/run_metrla.py data/external/metr-la.h5
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SEEDS = range(5)
RATE = 0.25


def load(path: str):
    d = pd.read_hdf(path)
    x = d.values.astype(np.float32)
    avail = x != 0.0                       # METR-LA marks missing with zero
    return x, avail


def locf(x, seen):
    """Last observed value per sensor, carried forward over the time axis."""
    out = np.empty_like(x)
    last = np.zeros(x.shape[1], dtype=np.float32)
    have = np.zeros(x.shape[1], dtype=bool)
    for t in range(x.shape[0]):
        out[t] = np.where(have, last, np.nan)
        upd = seen[t]
        last = np.where(upd, x[t], last)
        have |= upd
    return out


def lag(x, seen, k):
    """The sensor's own value k steps earlier, if it was observed then."""
    out = np.full_like(x, np.nan)
    out[k:] = np.where(seen[:-k], x[:-k], np.nan)
    return out


def main(path: str) -> None:
    x, avail = load(path)
    T = x.shape[0]
    test = slice(int(T * 0.8), T)
    print(f"METR-LA {x.shape}, missing {100 * (1 - avail.mean()):.1f}%, "
          f"test span {test.start}:{T}")
    rows = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        target = avail & (rng.random(x.shape) < RATE)      # what we must impute
        seen = avail & ~target                             # what the model sees
        preds = {
            "LOCF (previous step)": locf(x, seen),
            "Lag 1 day (288 steps)": lag(x, seen, 288),
            "Lag 1 week (2016 steps)": lag(x, seen, 2016),
        }
        # the MEAN baseline both papers do report, for scale
        mu = np.where(seen[:test.start].any(0),
                      np.nansum(np.where(seen[:test.start], x[:test.start], 0), 0)
                      / np.maximum(seen[:test.start].sum(0), 1), 0.0)
        preds["MEAN (per sensor)"] = np.broadcast_to(mu, x.shape)
        m = target[test]
        for k, p in preds.items():
            pt = p[test]
            ok = m & np.isfinite(pt)
            rows.setdefault(k, []).append(
                (float(np.abs(pt[ok] - x[test][ok]).mean()), float(ok.sum() / m.sum())))
    print(f"\n{'Baseline':<26}{'MAE':>8}{'coverage':>10}")
    for k, v in rows.items():
        mae = np.mean([a for a, _ in v]); cov = np.mean([c for _, c in v])
        print(f"{k:<26}{mae:>8.2f}{cov:>8.1%}")
    print(f"\nmean over {len(SEEDS)} seeds, injected rate {RATE:.0%}")
    print("\nValues reported by the GRIN paper (METR-LA point missing, MAE)")
    for k, v in [("KNN", 7.88), ("Mean", 7.56), ("MF", 5.56), ("MICE", 4.42),
                 ("VAR", 2.69), ("rGAIN", 2.83), ("MPGRU", 2.44),
                 ("BRITS", 2.34), ("GRIN", 1.91)]:
        print(f"  {k:<8}{v:>6.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
