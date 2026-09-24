"""Item 5: the first days after a transition, one day at a time.

Moving the shift onset to the first evaluation day and reporting one MAE over
the whole span averages the hard first fortnight away with the easy remainder.
This records the error per evaluation day and the cumulative error at day 1, 3,
7 and 14, for the two learned configurations and for both training-free
predictors. The common-mode baseline is the one that reads today's
observed nodes, so putting it on the same axis shows when a window mean computed
from before the transition stops being usable and how much a current reading
makes up for it.

Usage: CUDA_VISIBLE_DEVICES=1 python3 experiments/run_transition.py configs/cta.yaml
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
import kriging_data as kd
from kriging_data import make_dataset
from train import train_one

SCEN = "inregime"
RATIO = 0.5
ALPHA = 0.25
ONSET = "onset"
TRAIN_SEEDS = [0, 1]
MASK_SEEDS = [0, 1, 2]
HORIZON = 21


@torch.no_grad()
def per_day(model, bundle, obs, device):
    """Absolute error summed over hidden cells, per evaluation day."""
    model.eval()
    sc = bundle.scaler
    mean = torch.tensor(sc.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(sc.std_, dtype=torch.float32, device=device)
    ob_t = torch.tensor(obs.astype(np.float32), device=device)
    inner = getattr(model, "inner", model)
    bs = min(64, getattr(type(inner), "HP", {}).get("batch", 64))
    loader = DataLoader(make_dataset(bundle, bundle.test), batch_size=bs)
    err, cnt = [], []
    for hist, x_t, y_raw, valid in loader:
        hist, x_t, valid = hist.to(device), x_t.to(device), valid.to(device)
        ob = ob_t.expand(hist.size(0), -1) * valid
        out = model(hist, x_t * ob, ob)
        inv = (torch.expm1(out * std + mean).clamp_min(0) if sc.log1p
               else out * std + mean)
        m = ((1 - ob) * valid).bool()
        e = ((inv - y_raw.to(device)).abs() * m).sum(1)
        err.append(e.cpu().numpy()); cnt.append(m.sum(1).cpu().numpy())
    return np.concatenate(err), np.concatenate(cnt)


def free_per_day(bundle, obs, lag=7):
    raw = bundle.raw
    e1, e2, cn = [], [], []
    for t in bundle.test:
        gt, pr = raw[t], raw[t - lag]
        hid = (obs == 0) & ~np.isnan(gt) & ~np.isnan(pr)
        seen = (obs == 1) & ~np.isnan(gt) & ~np.isnan(pr)
        delta = (np.median(np.log1p(np.clip(gt[seen], 0, None))
                           - np.log1p(np.clip(pr[seen], 0, None)))
                 if seen.any() else 0.0)
        cm = np.expm1(np.log1p(np.clip(pr[hid], 0, None)) + delta).clip(0)
        e1.append(float(np.abs(gt[hid] - pr[hid]).sum()))
        e2.append(float(np.abs(gt[hid] - cm).sum()))
        cn.append(int(hid.sum()))
    return np.array(e1), np.array(e2), np.array(cn)


def main(config):
    cfg = utils.load_config(config)
    device = utils.get_device(cfg["train"]["device"])
    res = Path(cfg["paths"]["results"])
    name = Path(config).stem
    rows, t0 = [], time.time()

    for model_name in ("spin", "spin_c"):
        for ts in TRAIN_SEEDS:
            kd.set_level_shift(1.0, "pre")
            model, _, _, _ = train_one(cfg, SCEN, model_name, RATIO, "random", 0,
                                       graph_override="A_geo", verbose=False,
                                       train_seed=ts)
            kd.set_level_shift(ALPHA, ONSET)
            for ms in MASK_SEEDS:
                b = kd.load_bundle(cfg, SCEN, RATIO, "random", ms,
                                   graph_override="A_geo")
                e, c = per_day(model, b, b.observed, device)
                f1, f2, cn = free_per_day(b, b.observed)
                for d in range(min(HORIZON, len(e))):
                    rows.append({"dataset": name, "model": model_name,
                                 "train_seed": ts, "mask_seed": ms,
                                 "day": d + 1, "err": e[d], "n": c[d],
                                 "err_snaive7": f1[d], "err_cm": f2[d],
                                 "n_free": cn[d]})
            print(f"  {model_name} ts{ts} ({(time.time()-t0)/60:.1f} min)", flush=True)

    kd.set_level_shift(1.0, "pre")
    df = pd.DataFrame(rows)
    out = res / "transition_daily.csv"
    df.to_csv(out, index=False)
    print(f"\n[done] {len(df)} rows -> {out} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/cta.yaml")
