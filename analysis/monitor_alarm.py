"""The diagnostic turned into a rolling-window monitor.

Section 4.2 reports |shift| over a whole evaluation span, which is a
characterisation after the fact. An operator wants the same quantity on a
window that moves with the data. This script does that and reports how often it
fires, which is the number the appendix quotes.

Definition. The system level is the mean over observed nodes of the scaled
series, exactly as in the diagnostic. The monitor compares that level, averaged
over the trailing WINDOW days, against its mean over the training span. The
threshold is the largest deviation seen on the VALIDATION days, so it is set
without looking at the evaluation span; a day whose deviation exceeds it raises
an alarm.

Usage:  python3 src/monitor_alarm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathlib import Path

import numpy as np
import pandas as pd

import utils

WINDOW = 28
DATASETS = [("bike", "Bikeshare"), ("cta", "Chicago"),
            ("base", "Airports"), ("mta", "Subway")]


def alarms(cfg_path: str, scenario: str):
    cfg = utils.load_config(cfg_path)
    proc = Path(cfg["paths"]["processed"])
    m = pd.read_parquet(proc / "traffic_matrix.parquet")
    sp = utils.load_json(proc / "splits.json")["scenarios"][scenario]
    sc = utils.TrafficScaler.from_dict(sp["scaler"])
    level = np.nanmean(sc.transform(m.values), axis=1)
    roll = pd.Series(level).rolling(WINDOW, min_periods=1).mean().values

    tr, va, te = (np.asarray(sp[k]) for k in ("train", "val", "test"))
    base = float(np.nanmean(level[tr]))
    dev = np.abs(roll - base)
    thr = float(np.nanmax(dev[va]))                 # set without touching test
    return float(np.mean(dev[te] > thr) * 100)


def main():
    print(f"{'':<11}{'shock alarm %':>15}{'normal false alarm %':>23}")
    for conf, name in DATASETS:
        p = f"configs/{conf}.yaml"
        print(f"{name:<11}{alarms(p, 'cross_pre2shock'):>14.1f}%"
              f"{alarms(p, 'inregime'):>22.1f}%")


if __name__ == "__main__":
    main()
