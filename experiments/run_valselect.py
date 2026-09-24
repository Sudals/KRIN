"""Record validation MAE per (backbone, training seed) so the backbone can be
chosen without looking at the test split.

The paper reports SPIN + KRIN. It is fair to ask whether SPIN was
picked because it won on the test set. This script trains the same six centred
backbones under the same protocol and writes only the validation MAE that early
stopping already selects on, so the choice can be made on validation and the
test comparison read afterwards.

Usage:  CUDA_VISIBLE_DEVICES=2 python3 src/run_valselect.py configs/cta.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
import time
from pathlib import Path

import pandas as pd

import utils
from train import train_one

MODELS = ["ignnk_c", "satcn_c", "grin_c", "spin_c", "kits_c", "stgnn_c"]
SEEDS = [0, 1, 2]
SCEN = "cross_pre2shock"


def main(config: str) -> None:
    cfg = utils.load_config(config)
    res = Path(cfg["paths"]["results"])
    rows, t0 = [], time.time()
    for name in MODELS:
        for ts in SEEDS:
            _, _, _, best = train_one(cfg, SCEN, name, 0.5, "random", 0,
                                      graph_override="A_geo", verbose=False,
                                      train_seed=ts)
            rows.append({"model": name, "train_seed": ts, "val_MAE": best})
            print(f"  {name:<10} seed {ts}  valMAE {best:.2f}"
                  f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(res / "val_select.csv", index=False)
    m = df.groupby("model").val_MAE.mean().sort_values()
    print("\nmean validation MAE (lower is better)")
    for k, v in m.items():
        print(f"  {k:<10} {v:>10.2f}")
    print(f"\nbackbone chosen on validation: {m.index[0]}  ->  {res}/val_select.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml")
