"""Emit the results table for the released-implementation arm.

Prints the table as markdown so the numbers reported in the paper are generated
from the run files rather than transcribed. Datasets whose run has not finished
are skipped and listed at the end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DS = [("bike", "Bikeshare"), ("cta", "Chicago"),
      ("base", "Airports"), ("mta", "Subway")]
MODELS = [("SNaive-7", "`SNaive-7`"), ("ignnk_off", "IGNNK-R"), ("satcn_off", "SATCN-R"),
          ("grin_off", "GRIN-R"), ("kits_off", "KITS-R"), ("spin_off", "SPIN-R")]


def cell(v: float | None) -> str:
    return "-" if v is None or np.isnan(v) else f"{v:,.1f}"


def main() -> None:
    got, missing = {}, []
    for cfg, name in DS:
        res = yaml.safe_load(open(f"configs/{cfg}.yaml"))["paths"]["results"]
        f = Path(res) / "runs_official2.csv"
        if not f.exists():
            missing.append(name)
            continue
        d = pd.read_csv(f)
        d = d[(d.strategy == "random") & (np.isclose(d.ratio, 0.5)) & (d.regime == "all")]
        b = pd.read_csv(Path(res) / "baselines.csv")
        b = b[(b.strategy == "random") & (np.isclose(b.ratio, 0.5))
              & (b.model == "SNaive-7") & (b.regime == "all")]
        r = {}
        for sc in ("inregime", "cross_pre2shock"):
            r[("SNaive-7", sc)] = b[b.scenario == sc].MAE.mean()
            for m, _ in MODELS[1:]:
                for suf in ("", "_c"):
                    v = d[(d.model == m + suf) & (d.scenario == sc)].MAE.mean()
                    if not np.isnan(v):
                        r[(m + suf, sc)] = v
        got[name] = r

    names = [n for _, n in DS if n in got]
    print("| Model | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for key, label in MODELS:
        for suf, tag in (("", ""), ("_c", " + KRIN")):
            if key == "SNaive-7" and suf:
                continue
            cells = []
            for n in names:
                r = got[n]
                cells.append(f"{cell(r.get((key + suf, 'inregime')))} / "
                             f"{cell(r.get((key + suf, 'cross_pre2shock')))}")
            if all(c == "- / -" for c in cells):
                continue
            print(f"| {label}{tag} | " + " | ".join(cells) + " |")
    if missing:
        print(f"\n[incomplete] {', '.join(missing)}")


if __name__ == "__main__":
    main()
