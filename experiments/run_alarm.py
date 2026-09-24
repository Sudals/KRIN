"""Item 5b: detecting a shift is not the same as predicting a reversal.

The rolling monitor fires on a level move. What a practitioner wants to know is
something narrower: on this day, has the learned model actually fallen behind the
training-free baseline? This evaluates the two questions separately on the real
shock span, day by day, and reports the monitor's precision and recall against
the day-level reversal it is supposed to anticipate.

Usage: CUDA_VISIBLE_DEVICES=1 python3 experiments/run_alarm.py configs/cta.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import utils
import kriging_data as kd
from train import train_one
from run_transition import per_day, free_per_day

SCEN = "cross_pre2shock"
RATIO = 0.5
TRAIN_SEEDS = [0, 1]
MASK_SEEDS = [0, 1, 2]
WINDOW = 28


def monitor_alarms(cfg):
    """Per-evaluation-day alarm from the rolling-window monitor of A.2.10."""
    proc = Path(cfg["paths"]["processed"])
    m = pd.read_parquet(proc / "traffic_matrix.parquet")
    sp = utils.load_json(proc / "splits.json")["scenarios"][SCEN]
    sc = utils.TrafficScaler.from_dict(sp["scaler"])
    level = np.nanmean(sc.transform(m.values), axis=1)
    roll = pd.Series(level).rolling(WINDOW, min_periods=1).mean().values
    tr, va, te = (np.asarray(sp[k]) for k in ("train", "val", "test"))
    dev = np.abs(roll - float(np.nanmean(level[tr])))
    thr = float(np.nanmax(dev[va]))
    return dev[te] > thr


def main(config):
    cfg = utils.load_config(config)
    device = utils.get_device(cfg["train"]["device"])
    res = Path(cfg["paths"]["results"])
    name = Path(config).stem
    alarm = monitor_alarms(cfg)
    rows = []
    for model_name in ("spin", "spin_c"):
        for ts in TRAIN_SEEDS:
            model, _, _, _ = train_one(cfg, SCEN, model_name, RATIO, "random", 0,
                                       graph_override="A_geo", verbose=False,
                                       train_seed=ts)
            for ms in MASK_SEEDS:
                b = kd.load_bundle(cfg, SCEN, RATIO, "random", ms,
                                   graph_override="A_geo")
                e, c = per_day(model, b, b.observed, device)
                f1, f2, cn = free_per_day(b, b.observed)
                for d in range(len(e)):
                    rows.append({"dataset": name, "model": model_name,
                                 "train_seed": ts, "mask_seed": ms, "day": d,
                                 "alarm": bool(alarm[d]), "err": e[d], "n": c[d],
                                 "err_snaive7": f1[d], "err_cm": f2[d],
                                 "n_free": cn[d]})
            print(f"  {model_name} ts{ts} done", flush=True)
    df = pd.DataFrame(rows)
    out = res / "alarm_split.csv"
    df.to_csv(out, index=False)
    print(f"[done] {len(df)} rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/cta.yaml")
