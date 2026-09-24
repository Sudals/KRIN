"""Does an expressive aggregator really encode the absolute level more?

The paper's mechanism claim is that models which lose most under a level shift
are the ones whose learned components encode the absolute level. That is usually
argued from the performance drop itself, which is circular. This measures the
property directly and without touching any architecture.

Level equivariance. Add a constant c to every input (history and observed target
day, in the scaled space the model works in) and ask whether the prediction moves
by exactly c:

    gap(c) = mean | f(H+c, (x+c)*m, m) - ( f(H, x*m, m) + c ) | / c

  gap = 0  the model is level-equivariant; it reads only deviations, so a level
           shift cannot invalidate what it learned
  gap = 1  the prediction does not move at all with the level; the level is
           baked into the learned mapping

KRIN's wrapper is exactly equivariant by construction, so it must score 0. That
is the control: if it does not, the probe is wrong, not the model.

Usage: CUDA_VISIBLE_DEVICES=2 python3 src/run_probe.py configs/mta.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
from kriging_data import load_bundle, make_dataset
from train import train_one

MODELS = ["ignnk", "satcn", "grin", "spin", "kits", "stgnn", "spin_c"]
CS = [0.25, 0.5, 1.0]
SCEN = "inregime"
RATIO = 0.5
TRAIN_SEEDS = [0, 1, 2]


@torch.no_grad()
def equivariance_gap(model, bundle, device):
    model.eval()
    obs = torch.tensor(bundle.observed, dtype=torch.float32, device=device)
    loader = DataLoader(make_dataset(bundle, bundle.test), batch_size=64)
    out = {c: [] for c in CS}
    for hist, x_t, y_raw, valid in loader:
        hist, x_t, valid = hist.to(device), x_t.to(device), valid.to(device)
        ob = obs.expand(hist.size(0), -1) * valid
        hid = (1 - ob) * valid                       # score on hidden nodes only
        base = model(hist, x_t * ob, ob)
        for c in CS:
            sh = model(hist + c, (x_t + c) * ob, ob)
            d = (sh - base - c).abs() * hid
            out[c].append((d.sum() / hid.sum().clamp_min(1)).item() / c)
    return {c: float(np.mean(v)) for c, v in out.items()}


def main(config):
    cfg = utils.load_config(config)
    device = utils.get_device(cfg["train"]["device"])
    res = Path(cfg["paths"]["results"]); res.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for name in MODELS:
        for ts in TRAIN_SEEDS:
            model, bundle, _, _ = train_one(cfg, SCEN, name, RATIO, "random", 0,
                                            graph_override="A_geo", verbose=False,
                                            train_seed=ts)
            g = equivariance_gap(model, bundle, device)
            rows.append({"model": name, "train_seed": ts,
                         **{f"gap_c{c}": g[c] for c in CS}})
            print(f"  {name:8s} ts{ts}  " +
                  "  ".join(f"c={c}:{g[c]:.3f}" for c in CS) +
                  f"   ({(time.time()-t0)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    out = res / "probe_equivariance.csv"; df.to_csv(out, index=False)
    print("\n" + df.groupby("model").mean(numeric_only=True).round(3).to_string())
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml")
