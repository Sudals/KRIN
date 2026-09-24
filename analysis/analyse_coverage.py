"""Paired analysis of the coverage runs, done on runs rather than on averages.

Three quantities, each formed inside a run and only then aggregated:

  outside penalty      MAE(outside) - MAE(inside) at a fixed shift condition
  extra penalty        that penalty under the shift, minus the same penalty with
                       no shift; what the shift costs beyond the placement's own
                       difficulty
  KRIN reduction       the extra penalty without centring, minus with it

Uncertainty comes from a paired bootstrap over the (train seed, placement seed)
pairs, and the sign count is reported alongside it so a mixed direction cannot
hide behind an interval.

Usage: python3 analysis/analyse_coverage.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DS = [("results_cta", "Chicago"), ("results_mta", "Subway")]
KEY = ["train_seed", "placement_seed"]
B = 20000


def ci(x, rng, b=B):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(np.mean(x)), np.nan, np.nan
    draws = rng.choice(x, size=(b, len(x)), replace=True).mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def penalties(d, model):
    """Per-run outside penalty under each shift condition, for one model."""
    m = d[d.model == model]
    out = {}
    for sh in ("no shift", "regional shift"):
        s = m[m["shift_cond"] == sh]
        i = s[s.coverage == "inside"].set_index(KEY).MAE
        o = s[s.coverage == "outside"].set_index(KEY).MAE
        idx = i.index.intersection(o.index)
        out[sh] = (o[idx] - i[idx]).rename("pen")
    return out


def main():
    rng = np.random.default_rng(0)
    for res, name in DS:
        f = Path(res) / "coverage_runs.csv"
        if not f.exists():
            print(f"{name}: no run file"); continue
        d = pd.read_csv(f)
        # "shift" is also a DataFrame method, so attribute access on it
        # silently returns the method instead of the column.
        d = d.rename(columns={"shift": "shift_cond"})
        print(f"\n===== {name}  ({len(d)} runs, "
              f"training seeds {sorted(d.train_seed.unique())}, "
              f"placement seeds {sorted(d.placement_seed.unique())})")
        models = [m for m in d.model.unique()]
        pairs = {}
        for model in models:
            p = penalties(d, model)
            print(f"  [{model}]")
            for sh in ("no shift", "regional shift"):
                v = p[sh]
                mu, lo, hi = ci(v.values, rng)
                print(f"    {sh:<15} outside penalty {mu:>9,.1f} "
                      f"[{lo:>8,.1f}, {hi:>8,.1f}]  "
                      f"positive in {int((v.values > 0).sum())}/{len(v)}")
            extra = (p["regional shift"] - p["no shift"]).dropna()
            mu, lo, hi = ci(extra.values, rng)
            print(f"    extra penalty from the shift {mu:>9,.1f} "
                  f"[{lo:>8,.1f}, {hi:>8,.1f}]  "
                  f"positive in {int((extra.values > 0).sum())}/{len(extra)}")
            pairs[model] = extra
        base = [m for m in models if not m.endswith("_c")]
        for b_ in base:
            c_ = b_ + "_c"
            if c_ not in pairs:
                continue
            idx = pairs[b_].index.intersection(pairs[c_].index)
            red = (pairs[b_][idx] - pairs[c_][idx])
            mu, lo, hi = ci(red.values, rng)
            rel = 100 * red.values / np.maximum(pairs[b_][idx].values, 1e-9)
            rmu, rlo, rhi = ci(rel, rng)
            print(f"  extra penalty removed by KRIN ({b_}): {mu:>9,.1f} "
                  f"[{lo:>8,.1f}, {hi:>8,.1f}]  "
                  f"positive in {int((red.values > 0).sum())}/{len(red)}  "
                  f"relative {rmu:>6.1f}% [{rlo:.1f}, {rhi:.1f}]")
        # training-free references, which do not depend on the model
        r = d[d.model == models[0]]
        for sh in ("no shift", "regional shift"):
            s = r[r["shift_cond"] == sh]
            for col, lab in (("snaive7", "SNaive-7"), ("common_mode", "+CM")):
                i = s[s.coverage == "inside"].set_index(KEY)[col]
                o = s[s.coverage == "outside"].set_index(KEY)[col]
                idx = i.index.intersection(o.index)
                v = (o[idx] - i[idx]).values
                print(f"  [{lab}] {sh:<15} outside penalty {v.mean():>8,.1f} "
                      f"(inside {i[idx].mean():,.1f} → outside {o[idx].mean():,.1f})")


if __name__ == "__main__":
    main()
