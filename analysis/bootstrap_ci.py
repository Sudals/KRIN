"""Paired confidence intervals for the margins the paper reports.

A condition holds 30 evaluations: 3 training runs, each scored against 10 fixed
mask draws. Those 30 are not independent, and there are two defensible ways to
say so.

  mask      resample the 10 mask seeds as clusters, keeping all training seeds
            inside each. This treats the training runs as fixed and asks only
            how much the mask draw moves the margin.
  nested    resample the 3 training seeds first, then resample mask seeds inside
            each drawn training seed. This treats the training run as the
            sampling unit, which is the stricter reading and the one
            is likely to ask for. With only 3 training runs it is necessarily
            wide.

Both are computed on the SAME paired quantity: for every (train seed, mask seed)
cell, the relative difference between the two models, so shared variation from
the mask draw and from the training run cancels before resampling.

Usage:  python3 src/bootstrap_ci.py [B]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils                                            # noqa: E402

DATASETS = [("Bikeshare", "configs/bike.yaml", "all"),
            ("Subway", "configs/mta.yaml", "all"),
            ("Airports", "configs/base.yaml", "shock2020"),
            ("Chicago", "configs/cta.yaml", "all")]
CORE = ["ignnk", "satcn", "grin", "spin", "kits", "stgnn"]
SCEN = "cross_pre2shock"
RATIO = 0.5


def runs(cfg_path: str, regime: str) -> pd.DataFrame:
    res = Path(utils.load_config(cfg_path)["paths"]["results"])
    frames = []
    for f in ("runs_geo3m10.csv", "runs_ablate.csv", "runs_ablate2.csv"):
        if (res / f).exists():
            frames.append(pd.read_csv(res / f))
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.strategy == "random") & (np.isclose(d.ratio, RATIO))
          & (d.scenario == SCEN) & (d.regime == regime)]
    return d.drop_duplicates(["model", "train_seed", "mask_seed"])


def paired(d: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Relative gap of a against b, one row per (train seed, mask seed)."""
    ka = d[d.model == a].set_index(["train_seed", "mask_seed"]).MAE
    kb = d[d.model == b].set_index(["train_seed", "mask_seed"]).MAE
    idx = ka.index.intersection(kb.index)
    g = ((ka[idx] - kb[idx]) / kb[idx] * 100).rename("gap").reset_index()
    return g


def ci(g: pd.DataFrame, how: str, B: int, rng) -> tuple[float, float, float]:
    masks = np.sort(g.mask_seed.unique())
    trains = np.sort(g.train_seed.unique())
    by = {(t, m): v for t, m, v in g[["train_seed", "mask_seed", "gap"]].values}
    stat = []
    for _ in range(B):
        if how == "mask":
            drawn = rng.choice(masks, len(masks), replace=True)
            vals = [by[(t, m)] for m in drawn for t in trains if (t, m) in by]
        else:                                   # nested: training seed first
            vals = []
            for t in rng.choice(trains, len(trains), replace=True):
                for m in rng.choice(masks, len(masks), replace=True):
                    if (t, m) in by:
                        vals.append(by[(t, m)])
        if vals:
            stat.append(float(np.mean(vals)))
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(g.gap.mean()), float(lo), float(hi)


def main(B: int = 4000) -> None:
    rng = np.random.default_rng(0)
    print(f"Shock condition, r={RATIO}, paired relative gap (%), B={B}")
    print(f"{'':<11}{'comparison':<30}{'estimate':>10}"
          f"{'mask clusters':>22}{'training seeds first':>24}")
    for name, cfg, sreg in DATASETS:
        d = runs(cfg, sreg)
        best_un = min(CORE, key=lambda m: d[d.model == m].MAE.mean())
        # C compares against the best centred backbone OTHER than the one we
        # report, which is the runner-up once every backbone is centred
        best_c = min([m for m in CORE if m != "spin"],
                     key=lambda m: d[d.model == m + "_c"].MAE.mean())
        for label, a, b in (("A: vs best uncentred", "spin_c", best_un),
                            ("B: KRIN on same backbone", "spin_c", "spin"),
                            ("C: vs best centred", "spin_c", best_c + "_c")):
            g = paired(d, a, b)
            if g.empty or a == b:
                continue
            pt, l1, h1 = ci(g, "mask", B, rng)
            _, l2, h2 = ci(g, "nested", B, rng)
            print(f"{name:<11}{label:<26}{pt:>+8.1f}   [{l1:>+6.1f}, {h1:>+6.1f}]"
                  f"      [{l2:>+6.1f}, {h2:>+6.1f}]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
