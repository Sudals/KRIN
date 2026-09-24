"""Item 4: what observation does a level shift need in order to be recoverable?

The equivariance proposition says a centred model is invariant to a level shift
that is already inside its window. It says nothing about where the level is
supposed to come from when the window is stale, or when the shift is regional
rather than system-wide. Three sweeps vary exactly that, holding everything else
fixed, and read the answer off the relative standing of four predictors:

  SNaive-7        copies the node's own value from a week ago; no current data
  SNaive-7+CM     the same, corrected by today's median change over observed nodes
  SPIN            a spatial model with no centring
  SPIN + KRIN     the same model centred on the node's own K-day window

  (A) global vs regional   the same realised |shift|, moving every node or one
                           spatial cluster; alpha for the regional arm is solved
                           so the realised shift matches
  (B) coverage             the same hidden nodes and the same number of observed
                           nodes, with the observed set inside or outside the
                           region that moved
  (C) freshness            the same global shift, with the hidden nodes' own
                           history ending 0, 3, 7 or 14 days ago

Each model is trained ONCE per (dataset, seed) on unshifted data; every condition
is applied at evaluation only.

Usage: CUDA_VISIBLE_DEVICES=1 python3 experiments/run_history.py configs/cta.yaml
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

import utils
import kriging_data as kd
from evaluate import test_metrics
from train import train_one

SCEN = "inregime"
RATIO = 0.5
TRAIN_SEEDS = [0, 1]
MASK_SEEDS = [0, 1, 2]
ALPHA = 0.25                 # the collapse-inducing downward shift used in C.2
REGION_FRAC = 0.30           # share of nodes that move in the regional arm
# 13 leaves a single real slot at t-K and 14 leaves none, which separates
# 'the only observation left is old' from 'there is no observation left'.
STALE = [0, 3, 7, 10, 12, 13, 14]


# --------------------------------------------------------------------------- #
# training-free predictors, evaluated on the hidden nodes of a given mask
def _free(bundle, obs, lag=7):
    """(SNaive-7, SNaive-7+CM) MAE on hidden-and-valid cells."""
    raw = bundle.raw
    e1 = e2 = 0.0
    n = 0
    for t in bundle.test:
        gt, pr = raw[t], raw[t - lag]
        hid = (obs == 0) & ~np.isnan(gt) & ~np.isnan(pr)
        if not hid.any():
            continue
        # today's common change, read off the nodes that ARE reporting
        seen = (obs == 1) & ~np.isnan(gt) & ~np.isnan(pr)
        delta = (np.median(np.log1p(np.clip(gt[seen], 0, None))
                           - np.log1p(np.clip(pr[seen], 0, None)))
                 if seen.any() else 0.0)
        cm = np.expm1(np.log1p(np.clip(pr[hid], 0, None)) + delta).clip(0)
        e1 += float(np.abs(gt[hid] - pr[hid]).sum())
        e2 += float(np.abs(gt[hid] - cm).sum())
        n += int(hid.sum())
    return e1 / max(n, 1), e2 / max(n, 1)


def realised_shift(cfg, alpha, nodes=None):
    """|shift| the sweep actually produces, in the diagnostic's own units."""
    proc = Path(cfg["paths"]["processed"])
    raw = pd.read_parquet(proc / "traffic_matrix.parquet").values.astype(float)
    sp = utils.load_json(proc / "splits.json")
    sc = sp["scenarios"][SCEN]
    scaler = utils.TrafficScaler.from_dict(sc["scaler"])
    tr = np.asarray(sc["train"] + sc["val"])
    te = np.asarray(sc["test"])
    lvl_tr = np.nanmean(scaler.transform(raw), axis=1)[tr]
    sh = raw.copy()
    b0 = max(0, min(sc["test"]) - sp["window_K"])
    if nodes is None:
        sh[b0:] *= alpha
    else:
        sh[np.ix_(np.arange(b0, sh.shape[0]), nodes)] *= alpha
    lvl_te = np.nanmean(scaler.transform(sh), axis=1)[te]
    return abs(float(np.nanmean(lvl_te) - np.nanmean(lvl_tr)))


def solve_alpha(cfg, target, nodes):
    """alpha whose realised |shift| matches ``target``.

    The realised shift falls monotonically as alpha rises toward 1, so the search
    walks that direction. Matching is done the other way round from the obvious
    one: a region holding 30% of the nodes cannot reach the shift a whole-panel
    alpha produces, while the whole panel can always reach the region's, so the
    regional arm sets the target and the global arm is solved to it.
    """
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if realised_shift(cfg, mid, nodes) > target:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def region(cfg, frac):
    """A spatially contiguous block: the ``frac`` nodes closest to one centre."""
    proc = Path(cfg["paths"]["processed"])
    A = np.load(proc / "A_geo.npy")
    N = A.shape[0]
    k = max(1, int(round(frac * N)))
    seed_node = int(np.argmax(A.sum(1)))          # the best-connected node
    order = np.argsort(-A[seed_node])             # nearest first by kernel weight
    return np.sort(order[:k])



@torch.no_grad()
def eval_on(model, bundle, obs, score, device):
    """MAE over ``score`` cells only, with ``obs`` as the observed input mask.

    ``test_metrics`` scores every cell the input mask hides, which is the right
    thing everywhere else. Experiment (B) moves the observed set around while the
    cells being scored have to stay fixed, so the two masks must be separate.
    """
    model.eval()
    sc = bundle.scaler
    mean = torch.tensor(sc.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(sc.std_, dtype=torch.float32, device=device)
    ob_t = torch.tensor(obs.astype(np.float32), device=device)
    inner = getattr(model, "inner", model)
    bs = min(64, getattr(type(inner), "HP", {}).get("batch", 64))
    from torch.utils.data import DataLoader
    from kriging_data import make_dataset
    loader = DataLoader(make_dataset(bundle, bundle.test), batch_size=bs)
    err, n = 0.0, 0
    sm = torch.tensor(score.astype(bool), device=device)
    for hist, x_t, y_raw, valid in loader:
        hist, x_t, valid = hist.to(device), x_t.to(device), valid.to(device)
        ob = ob_t.expand(hist.size(0), -1) * valid
        out = model(hist, x_t * ob, ob)
        inv = (torch.expm1(out * std + mean).clamp_min(0) if sc.log1p
               else out * std + mean)
        m = sm.expand(hist.size(0), -1) & (valid > 0)
        err += float((inv - y_raw.to(device)).abs()[m].sum())
        n += int(m.sum())
    return err / max(n, 1)


def free_on(bundle, obs, score, lag=7):
    """The two training-free predictors, scored on ``score`` only."""
    raw = bundle.raw
    e1 = e2 = 0.0
    n = 0
    for t in bundle.test:
        gt, pr = raw[t], raw[t - lag]
        hid = score & ~np.isnan(gt) & ~np.isnan(pr)
        if not hid.any():
            continue
        seen = (obs == 1) & ~np.isnan(gt) & ~np.isnan(pr)
        delta = (np.median(np.log1p(np.clip(gt[seen], 0, None))
                           - np.log1p(np.clip(pr[seen], 0, None)))
                 if seen.any() else 0.0)
        cm = np.expm1(np.log1p(np.clip(pr[hid], 0, None)) + delta).clip(0)
        e1 += float(np.abs(gt[hid] - pr[hid]).sum())
        e2 += float(np.abs(gt[hid] - cm).sum())
        n += int(hid.sum())
    return e1 / max(n, 1), e2 / max(n, 1)


# --------------------------------------------------------------------------- #
def main(config, out_tag="infoexp"):
    cfg = utils.load_config(config)
    device = utils.get_device(cfg["train"]["device"])
    proc = Path(cfg["paths"]["processed"])
    regime = np.array(utils.load_json(proc / "dates.json")["regime"])
    res = Path(cfg["paths"]["results"])
    name = Path(config).stem

    R = region(cfg, REGION_FRAC)
    a_reg = ALPHA
    tgt = realised_shift(cfg, a_reg, R)
    a_glob = solve_alpha(cfg, tgt, None)
    print(f"[info] {name}: regional alpha {a_reg} on {len(R)} of "
          f"{np.load(proc / 'A_geo.npy').shape[0]} nodes gives |shift| {tgt:.3f}; "
          f"global alpha solved to {a_glob:.4f} "
          f"(realised {realised_shift(cfg, a_glob, None):.3f})", flush=True)

    rows, t0 = [], time.time()
    for model_name in ("spin", "spin_c"):
        for ts in TRAIN_SEEDS:
            kd.set_level_shift(1.0, "pre"); kd.set_history_staleness(0)
            model, _, _, _ = train_one(cfg, SCEN, model_name, RATIO, "random", 0,
                                       graph_override="A_geo", verbose=False,
                                       train_seed=ts)

            def record(exp, cond, obs_override=None):
                for ms in MASK_SEEDS:
                    b = kd.load_bundle(cfg, SCEN, RATIO, "random", ms,
                                       graph_override="A_geo")
                    obs = b.observed if obs_override is None else obs_override(b)
                    m = test_metrics(model, b, obs, regime, device)["all"]
                    sn, cm = _free(b, obs)
                    rows.append({"dataset": name, "exp": exp, "cond": cond,
                                 "model": model_name, "train_seed": ts,
                                 "mask_seed": ms, "MAE": m["MAE"],
                                 "NMAE_macro": m.get("NMAE_macro", np.nan),
                                 "snaive7": sn, "common_mode": cm,
                                 "n_hidden": int((obs == 0).sum())})

            # (A) global vs regional, matched realised shift
            kd.set_level_shift(1.0, "pre"); record("A", "no shift")
            kd.set_level_shift(a_glob, "pre", None); record("A", "global")
            kd.set_level_shift(a_reg, "pre", R); record("A", "regional")

            # (B) coverage: the scored nodes are a fixed half of the moved
            #     region; the observed set has the same size in both arms and
            #     sits either inside the region or entirely outside it.
            kd.set_level_shift(a_reg, "pre", R)
            rng = np.random.default_rng(1234)
            Nn = np.load(proc / "A_geo.npy").shape[0]
            perm = rng.permutation(R)
            score = np.zeros(Nn, dtype=bool)
            score[perm[:max(1, len(R) // 2)]] = True
            pool_in = np.array([i for i in R if not score[i]])
            pool_out = np.array([i for i in range(Nn) if i not in set(R.tolist())])
            n_obs = int(min(len(pool_in), len(pool_out)))
            for cond, pool in (("observe inside the moved region", pool_in),
                               ("observe outside it", pool_out)):
                obs = np.zeros(Nn, dtype=bool)
                obs[rng.permutation(pool)[:n_obs]] = True
                for ms in MASK_SEEDS:
                    b = kd.load_bundle(cfg, SCEN, RATIO, "random", ms,
                                       graph_override="A_geo")
                    mae = eval_on(model, b, obs, score, device)
                    sn, cm = free_on(b, obs, score)
                    rows.append({"dataset": name, "exp": "B", "cond": cond,
                                 "model": model_name, "train_seed": ts,
                                 "mask_seed": ms, "MAE": mae,
                                 "NMAE_macro": np.nan, "snaive7": sn,
                                 "common_mode": cm,
                                 "n_hidden": int(score.sum()),
                                 "n_observed": n_obs})

            # (C) freshness of the hidden nodes' own history
            kd.set_level_shift(a_glob, "pre", None)
            for d in STALE:
                kd.set_history_staleness(d)
                record("C", f"stale {d}d")
            kd.set_history_staleness(0)
            print(f"  {model_name} ts{ts} done ({(time.time()-t0)/60:.1f} min)",
                  flush=True)

    kd.set_level_shift(1.0, "pre"); kd.set_history_staleness(0)
    df = pd.DataFrame(rows)
    out = res / f"{out_tag}.csv"
    df.to_csv(out, index=False)
    print(f"\n[done] {len(df)} rows -> {out} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/cta.yaml")
