"""Coverage experiment with a no-shift control, at run level.

The placement of the observed set is a random factor of its own, drawn afresh
for each placement seed, so replication counts the placements as well as the
training runs. Three factors vary:

  targets      a fixed half of the region that moves, identical in every cell
  coverage     the observed set, same size in both arms, drawn either from the
               rest of that region (inside) or from outside it
  shift        none, or the controlled regional shift applied to that region

Holding targets and both observed sets fixed across the shift condition is the
point. It separates "this placement is simply harder to predict from" from "this
placement loses information the shift made necessary": the first shows up in the
no-shift row, the second only in the difference between the rows.

Every row is one (model, train seed, placement seed, shift, coverage) run, so
pairing happens on the runs themselves rather than on their averages.

Usage: CUDA_VISIBLE_DEVICES=1 python3 experiments/run_coverage.py configs/cta.yaml spin,spin_c
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import utils
import kriging_data as kd
from train import train_one
from run_history import eval_on, free_on, region, realised_shift, REGION_FRAC, ALPHA

SCEN = "inregime"
RATIO = 0.5
TRAIN_SEEDS = [0, 1, 2]
PLACEMENT_SEEDS = [0, 1, 2, 3, 4]


def placement(cfg, R, N, p):
    """Targets and the two equal-sized observed sets for placement seed ``p``."""
    rng = np.random.default_rng(1000 + p)
    perm = rng.permutation(R)
    score = np.zeros(N, dtype=bool)
    score[perm[:max(1, len(R) // 2)]] = True
    pool_in = np.array([i for i in R if not score[i]])
    pool_out = np.array([i for i in range(N) if i not in set(R.tolist())])
    n_obs = int(min(len(pool_in), len(pool_out)))
    obs = {}
    for tag, pool in (("inside", pool_in), ("outside", pool_out)):
        o = np.zeros(N, dtype=bool)
        o[rng.permutation(pool)[:n_obs]] = True
        obs[tag] = o
    return score, obs, n_obs


def main(config, models):
    cfg = utils.load_config(config)
    device = utils.get_device(cfg["train"]["device"])
    res = Path(cfg["paths"]["results"])
    name = Path(config).stem
    N = np.load(Path(cfg["paths"]["processed"]) / "A_geo.npy").shape[0]

    R = region(cfg, REGION_FRAC)
    realised = realised_shift(cfg, ALPHA, R)
    print(f"[coverage] {name}: region {len(R)}/{N} nodes, alpha {ALPHA}, "
          f"realised |shift| {realised:.3f}", flush=True)

    plans = {p: placement(cfg, R, N, p) for p in PLACEMENT_SEEDS}
    rows, t0 = [], time.time()
    for model_name in models:
        for ts in TRAIN_SEEDS:
            kd.set_level_shift(1.0, "pre")
            kd.set_history_staleness(0)
            model, _, _, _ = train_one(cfg, SCEN, model_name, RATIO, "random", 0,
                                       graph_override="A_geo", verbose=False,
                                       train_seed=ts)
            for shift_tag, alpha in (("no shift", 1.0), ("regional shift", ALPHA)):
                if alpha == 1.0:
                    kd.set_level_shift(1.0, "pre")
                else:
                    kd.set_level_shift(alpha, "pre", R)
                b = kd.load_bundle(cfg, SCEN, RATIO, "random", 0,
                                   graph_override="A_geo")
                for p, (score, obs, n_obs) in plans.items():
                    for cov in ("inside", "outside"):
                        mae = eval_on(model, b, obs[cov], score, device)
                        sn, cm = free_on(b, obs[cov], score)
                        rows.append({"dataset": name, "model": model_name,
                                     "train_seed": ts, "placement_seed": p,
                                     "shift": shift_tag, "coverage": cov,
                                     "MAE": mae, "snaive7": sn,
                                     "common_mode": cm,
                                     "n_target": int(score.sum()),
                                     "n_observed": n_obs,
                                     "realised_shift": realised})
            print(f"  {model_name} ts{ts} ({(time.time()-t0)/60:.1f} min)", flush=True)

    kd.set_level_shift(1.0, "pre")
    out = res / "coverage_runs.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.model.isin(models)], pd.DataFrame(rows)],
                       ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\n[done] {len(rows)} new rows -> {out} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    a = sys.argv[1:]
    cfgp = a[0] if a else "configs/cta.yaml"
    ms = a[1].split(",") if len(a) > 1 else ["spin", "spin_c"]
    main(cfgp, ms)
