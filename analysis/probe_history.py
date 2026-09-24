"""Item 1: what the code actually hands a model as history, day by day.

For one node and one fixed mask, this dumps the exact history vector and mask
the model receives on days t, t+1, t+7 and t+14. It settles from the code what
the protocol does with a value hidden on day t: whether it returns as history on
later days, and what the window holds once holes are injected.

Usage:  python3 analysis/probe_history.py [config] [--block]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import utils
import kriging_data as kd
from kriging_data import load_bundle, make_dataset


def probe(cfg_path, block=False, rate=0.0):
    cfg = utils.load_config(cfg_path)
    if rate > 0:
        kd.set_hist_missing(rate, "block" if block else "point", 0, 7)
    b = load_bundle(cfg, "cross_pre2shock", 0.5, "random", 0, graph_override="A_geo")
    ds = make_dataset(b, b.test)
    hidden = np.flatnonzero(b.observed == 0)
    j = int(hidden[0])                       # a node hidden on every evaluation day
    print(f"\n[{Path(cfg_path).stem}] hist_missing={rate} block={block}")
    print(f"  hidden node j={j} ({b.nodes[j]}), {len(b.test)} evaluation days, K={b.K}")
    print(f"  is this node zero in the observed mask: {b.observed[j] == 0}")
    t0 = 0
    for off in (0, 1, 7, 14):
        if t0 + off >= len(ds):
            continue
        hist, x_t, y_raw, valid = ds[t0 + off]
        day = b.test[t0 + off]
        h = hist.numpy()[:, j]
        print(f"  t0+{off:<3} (row {day}): last 5 history slots {np.round(h[-5:], 3)}"
              f"  target-day truth {float(y_raw[j]):.1f}  valid={int(valid[j])}")
    # does the day-t0 value reappear inside the day-(t0+1) history?
    h1 = ds[t0 + 1][0].numpy()[:, j]
    z_t0 = b.scaled[b.test[t0], j]
    print(f"  does t0's scaled value {z_t0:.4f} equal the last slot of t0+1 history {h1[-1]:.4f}: "
          f"{np.isclose(z_t0, h1[-1])}")
    return b, j


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    cfgp = a[0] if a else "configs/cta.yaml"
    probe(cfgp, rate=0.0)
    probe(cfgp, block=False, rate=0.25)
    probe(cfgp, block=True, rate=0.25)
