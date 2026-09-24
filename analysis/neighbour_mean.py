"""What centring does to a weighted neighbour mean.

The point of the table is to separate two things that are easy to confuse. Part
of the gain from working in a normalised space comes from the GLOBAL log1p and
z-scaling that every model here uses anyway; the rest comes from subtracting each
node's own K-day window level. Running the same aggregator -- identical graph,
identical weights, identical hidden set -- in all three spaces isolates the
second part, which is the operation this paper is about.

Shock condition, r=0.5, averaged over the configured mask seeds. No training is
involved, so there is nothing to seed beyond the mask.

Usage:  python3 src/neighbour_mean.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import utils
from kriging_data import load_bundle

DATASETS = [("base", "Airports"), ("cta", "Chicago"),
            ("mta", "Subway"), ("bike", "Bikeshare")]


def three_spaces(cfg, seed):
    b = load_bundle(cfg, "cross_pre2shock", 0.5, "random", seed,
                    graph_override="A_geo")
    A = b.A_raw.copy()
    np.fill_diagonal(A, 0.0)                    # a hidden node cannot see itself
    obs = b.observed.astype(float)
    te, K, raw, sc = np.array(b.test), b.K, b.raw, b.scaler
    valid = ~np.isnan(raw)
    W = A * obs[None, :]                        # only observed neighbours count
    den = W.sum(1)
    den[den == 0] = np.nan
    hid = (obs == 0) & valid[te]
    Y = raw[te]

    inv = (lambda z: np.expm1(z * sc.std_ + sc.mean_).clip(0)) if sc.log1p \
        else (lambda z: z * sc.std_ + sc.mean_)

    p_raw = (np.nan_to_num(Y) * valid[te]) @ W.T / den[None, :]

    Z = sc.transform(raw)
    p_glob = inv(np.nan_to_num(Z[te]) * valid[te] @ W.T / den[None, :])

    # Per-node level: the mean of the node's own K-day window, observed days only.
    mu = np.stack([np.nanmean(np.where(valid[t - K:t], Z[t - K:t], np.nan), axis=0)
                   for t in te])
    mu = np.nan_to_num(mu)
    dev = (np.nan_to_num(Z[te]) - mu) * valid[te]
    p_cent = inv(dev @ W.T / den[None, :] + mu)

    return [utils.metrics(Y, p, hid)["MAE"] for p in (p_raw, p_glob, p_cent)]


def main():
    print(f"{'':<11}{'Raw':>11}{'Global':>11}{'+centring':>12}{'effect':>9}")
    for conf, name in DATASETS:
        cfg = utils.load_config(f"configs/{conf}.yaml")
        m = np.nanmean([three_spaces(cfg, s) for s in cfg["mask"]["seeds"]], axis=0)
        print(f"{name:<11}{m[0]:>11,.1f}{m[1]:>11,.1f}{m[2]:>12,.1f}"
              f"{m[1] / m[2]:>8.1f}x")


if __name__ == "__main__":
    main()
