"""Controlled level-shift sweep: does the diagnostic's DENOMINATOR earn its place?

Multiplying the evaluation span by alpha moves the numerator of shift/amplitude
directly, so "larger shift breaks it" is close to a tautology. What is not forced
is whether the collapse onset lines up ACROSS DATASETS once the shift is divided
by each dataset's own training-time amplitude. The four amplitudes differ by 2x
(subway 2.713 vs Chicago 5.388), so the two coordinate systems make different
predictions and only one of them can align.

Each model is trained ONCE on unshifted data; alpha is applied at evaluation
only, so the sweep costs one training per (model, seed).

Two modes (Appendix C.2):
  --mode=pre    main sweep (Tables 37-38): 12 alphas in [0.15, 5.0]; the shift
                covers the evaluation span AND the K days before it, so every
                history window is already at the new level -> shiftsweep.csv
  --mode=onset  onset variant (Table 39): alpha in {0.25, 0.5, 2.0}; the shift
                starts on the first evaluation day -> shiftsweep_onset.csv

Usage:
    python3 experiments/run_shift_sweep.py configs/mta.yaml --mode=pre
    python3 experiments/run_shift_sweep.py configs/mta.yaml --mode=onset
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import utils
import kriging_data as kd
from evaluate import test_metrics
from train import train_one

MODELS = ["ignnk", "satcn", "grin", "spin", "kits", "stgnn",
          "ignnk_c", "satcn_c", "grin_c", "spin_c", "kits_c", "stgnn_c"]
# Main sweep ("pre"): the K days before the span move too, so every test day
# sees a fully shifted history. Onset variant ("onset"): the shift starts exactly
# at the test span, so the first K evaluation days meet a history that straddles
# the transition.
MODES = {
    "pre":   dict(alphas=[0.15, 0.25, 0.35, 0.5, 0.7, 0.85,
                          1.0, 1.2, 1.5, 2.0, 3.0, 5.0],
                  out="shiftsweep.csv"),
    "onset": dict(alphas=[0.25, 0.5, 2.0], out="shiftsweep_onset.csv"),
}
SCEN = "inregime"
TRAIN_SEEDS = [0, 1, 2]
MASK_SEEDS = [0, 1, 2]
RATIO = 0.5


def diagnostic(cfg, alpha, onset="pre"):
    """shift/amp for a given alpha.

    The denominator must come from the UNSHIFTED matrix. Training always runs at
    alpha=1, so the level variation the model actually saw is a property of
    training and cannot depend on the evaluation shift. Computing amp on the
    shifted matrix let the K transition days before the test span fall inside the
    train+val rows and moved amp from 3.568 to 4.674 across the sweep, i.e. the
    side effect landed on the very quantity under test.
    """
    proc = Path(cfg["paths"]["processed"])
    raw = pd.read_parquet(proc / "traffic_matrix.parquet").values.astype(float)
    sp = utils.load_json(proc / "splits.json")
    sc = sp["scenarios"][SCEN]
    scaler = utils.TrafficScaler.from_dict(sc["scaler"])
    tr = np.asarray(sc["train"] + sc["val"])
    te = np.asarray(sc["test"])

    lvl_tr = np.nanmean(scaler.transform(raw), axis=1)[tr]
    shifted = raw.copy()
    b0 = min(sc["test"]) - (sp["window_K"] if onset == "pre" else 0)
    shifted[max(0, b0):] *= alpha
    lvl_te = np.nanmean(scaler.transform(shifted), axis=1)[te]

    shift = float(np.nanmean(lvl_te) - np.nanmean(lvl_tr))
    amp = float(np.nanmax(lvl_tr) - np.nanmin(lvl_tr))
    return shift, amp


def snaive7(bundle):
    """Training-free reference on the shifted data, hidden nodes only."""
    raw, obs = bundle.raw, bundle.observed
    err, n = 0.0, 0
    for t in bundle.test:
        gt, pr = raw[t], raw[t - 7]
        m = (~obs) & ~np.isnan(gt) & ~np.isnan(pr)
        err += float(np.abs(gt[m] - pr[m]).sum()); n += int(m.sum())
    return err / max(n, 1)


def main(config, mode="pre"):
    ALPHAS, ONSET, OUTNAME = MODES[mode]["alphas"], mode, MODES[mode]["out"]
    cfg = utils.load_config(config)
    device = utils.get_device(cfg["train"]["device"])
    proc = Path(cfg["paths"]["processed"])
    regime = np.array(utils.load_json(proc / "dates.json")["regime"])
    res = Path(cfg["paths"]["results"]); res.mkdir(parents=True, exist_ok=True)

    diag = {a: diagnostic(cfg, a, ONSET) for a in ALPHAS}
    print(f"[sweep] {config}  mode={mode}  realised shift/amp per alpha", flush=True)
    for a, (s, amp) in diag.items():
        print(f"    alpha={a:<5} shift={s:+.3f}  amp={amp:.3f}  |shift|/amp={abs(s)/amp:.3f}")

    rows, t0 = [], time.time()
    for name in MODELS:
        for ts in TRAIN_SEEDS:
            kd.set_level_shift(1.0, "pre")                     # train unshifted
            model, _, _, _ = train_one(cfg, SCEN, name, RATIO, "random", 0,
                                       graph_override="A_geo", verbose=False,
                                       train_seed=ts)
            for a in ALPHAS:
                kd.set_level_shift(a, ONSET)
                for ms in MASK_SEEDS:
                    b = kd.load_bundle(cfg, SCEN, RATIO, "random", ms,
                                       graph_override="A_geo")
                    m = test_metrics(model, b, b.observed, regime, device)["all"]
                    sh, amp = diag[a]
                    rows.append({"model": name, "alpha": a, "train_seed": ts,
                                 "mask_seed": ms, "MAE": m["MAE"],
                                 "NMAE_macro": m.get("NMAE_macro", np.nan),
                                 "snaive7": snaive7(b),
                                 "onset": ONSET,
                                 "shift": sh, "amp": amp,
                                 "shift_over_amp": abs(sh) / amp})
            print(f"  {name} ts{ts}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    kd.set_level_shift(1.0, "pre")
    df = pd.DataFrame(rows)
    out = res / OUTNAME; df.to_csv(out, index=False)
    print(f"\n[done] {len(df)} rows -> {out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:]
                if a.startswith("--") and "=" in a)
    mode = opts.get("mode", "pre")
    if mode not in MODES:
        raise SystemExit(f"--mode must be one of {sorted(MODES)}")
    main(args[0] if args else "configs/base.yaml", mode)
