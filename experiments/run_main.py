"""Final protocol: GEO graph, pre-vs-shock only, 3 TRAINING seeds x 3 mask seeds.

Differences from the earlier runs, all requested by the user:
  * graph  — always ``A_geo`` (no correlation component anywhere)
  * scenarios — ``inregime`` (normal) and ``cross_pre2shock`` only; the
    ``recovery`` regime is retired and simply skipped here, so splits.json is
    left intact for anything that still wants it.
  * seeds  — each (model, scenario, ratio) is TRAINED 3 times (train_seed
    0/1/2) and each trained model is evaluated on the 3 fixed mask seeds.
    Reported mean/std therefore span 9 evaluations and capture BOTH training
    randomness and mask variation; the old runs trained once and only varied
    the mask.

Writes two files into the config's results dir:
    runs_geo3.csv          long format, one row per (…, train_seed, mask_seed)
    main_results_geo3.csv  aggregated mean/std, same schema as main_results.csv

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 python3 experiments/run_main.py configs/base.yaml
    CUDA_VISIBLE_DEVICES=1 python3 experiments/run_main.py configs/mta.yaml --graph=A_mixed
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import utils
from evaluate import test_metrics
from kriging_data import load_bundle
from train import train_one

SCENARIOS = ["inregime", "cross_pre2shock"]
TRAIN_SEEDS = [0, 1, 2]
ALL_MODELS = ["gru", "stgnn", "ignnk", "satcn", "grin", "spin", "kits",
              "stgnnr", "krin"]


def main(models, config, graph="A_geo", mask_seeds=None, tag_override=None,
         scenarios_override=None, ratios_override=None, train_seeds=None,
         split_shock_years=False, hist_missing=0.0, hist_pattern="point",
         hist_seed=0):
    # "default" -> whatever splits.json says for the scenario. That matters for
    # leakage: the mixed graph is A_mixed in-regime but A_mixed_pre for the cross
    # scenarios, because its correlation half must be fit on the train span only.
    # Forcing a literal "A_mixed" everywhere would leak shock-period correlations.
    GRAPH = None if graph == "default" else graph
    tag = {"A_geo": "geo3", "default": "mixed3"}.get(graph, graph.replace("A_", "") + "3")
    tag = tag_override or tag
    if hist_missing > 0.0:
        # Protocol-B holes. Set BEFORE any bundle is built so training and
        # evaluation see the same holey matrix, identically for every model.
        import kriging_data as _kd
        _kd.set_hist_missing(hist_missing, hist_pattern, hist_seed)
        print(f"[final] history holes: rate={hist_missing} "
              f"pattern={hist_pattern} seed={hist_seed}", flush=True)
    cfg = utils.load_config(config)
    device = utils.get_device(cfg["train"]["device"])
    proc = Path(cfg["paths"]["processed"])
    res = Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)

    regime_arr = np.array(utils.load_json(proc / "dates.json")["regime"])
    if split_shock_years:
        # Airports is the only dataset whose shock regime spans TWO calendar years
        # (2020-21; the other three end in 2020), and its 2021 second half is back
        # to roughly normal levels. Reporting one number over 24 months therefore
        # is not comparable with the others' 12. This does NOT change what is
        # trained or evaluated -- it only partitions the same test days so the
        # 12-month-matched slice can be read alongside the full window.
        yrs = pd.to_datetime(utils.load_json(proc / "dates.json")["dates"]).year
        regime_arr = np.array([f"{r}{y}" if r == "shock" else r
                               for r, y in zip(regime_arr, yrs)])
        print(f"[final] shock split by year -> "
              f"{sorted(set(regime_arr[np.char.startswith(regime_arr.astype(str), 'shock')]))}")
    have = set(utils.load_json(proc / "splits.json")["scenarios"])
    scenarios = [s for s in (scenarios_override or SCENARIOS) if s in have]
    ratios = ratios_override or cfg["mask"]["ratios"]
    mask_seeds = mask_seeds or cfg["mask"]["seeds"]
    strategies = cfg["mask"]["strategies"]
    tseeds = train_seeds or TRAIN_SEEDS
    total = len(models) * len(scenarios) * len(ratios) * len(tseeds)
    print(f"[final] {config} device={device} graph={GRAPH or 'scenario default (mixed)'}")
    print(f"[final] models={models} scenarios={scenarios} ratios={ratios}")
    print(f"[final] train_seeds={tseeds} mask_seeds={mask_seeds} "
          f"-> {total} training runs", flush=True)

    t0, done, rows = time.time(), 0, []
    for name, scenario, ratio, tseed in itertools.product(
            models, scenarios, ratios, tseeds):
        done += 1
        print(f"\n=== [{done}/{total}] TRAIN {name} | {scenario} | r={ratio} "
              f"| train_seed={tseed} ===", flush=True)
        model, _, _, _ = train_one(cfg, scenario, name, ratio, "random", 0,
                                   graph_override=GRAPH, verbose=False,
                                   train_seed=tseed)
        for strategy in strategies:
            for mseed in mask_seeds:
                b = load_bundle(cfg, scenario, ratio, strategy, mseed,
                                graph_override=GRAPH)
                per = test_metrics(model, b, b.observed, regime_arr, device)
                for reg, met in per.items():
                    rows.append({"model": name, "scenario": scenario,
                                 "ratio": ratio, "strategy": strategy,
                                 "regime": reg, "train_seed": tseed,
                                 "mask_seed": mseed,
                                 **{k: v for k, v in met.items() if k != "n"}})
        cur = pd.DataFrame(rows)
        sel = cur[(cur.model == name) & (cur.scenario == scenario)
                  & (cur.ratio == ratio) & (cur.train_seed == tseed)
                  & (cur.strategy == "random") & (cur.regime == "all")]
        print(f"  MAE={sel.MAE.mean():.2f} R2={sel.R2.mean():.3f} "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

    long = pd.DataFrame(rows)
    out_long = res / f"runs_{tag}.csv"
    if out_long.exists():
        old = pd.read_csv(out_long)
        long = pd.concat([old[~old.model.isin(models)], long], ignore_index=True)
    long.to_csv(out_long, index=False)

    metrics = [c for c in long.columns if c not in
               ("model", "scenario", "ratio", "strategy", "regime",
                "train_seed", "mask_seed")]
    agg = (long.groupby(["model", "scenario", "ratio", "strategy", "regime"])[metrics]
           .agg(["mean", "std"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(res / f"main_results_{tag}.csv", index=False)

    print(f"\n[final] {total} training runs in {(time.time()-t0)/60:.1f} min")
    piv = agg[(agg.regime == "all") & (agg.strategy == "random")].pivot_table(
        index=["scenario", "ratio"], columns="model", values="MAE_mean").round(2)
    print(f"\n=== FINAL {config} | geo | MAE, mean over 3 train x 3 mask seeds ===")
    print(piv.to_string())
    print(f"\n[done] -> {out_long} , {res}/main_results_{tag}.csv")


if __name__ == "__main__":
    argv = sys.argv[1:]
    conf = argv[0] if argv and argv[0].endswith((".yaml", ".yml")) else "configs/base.yaml"
    argv = argv[1:] if argv and argv[0].endswith((".yaml", ".yml")) else argv
    graph, mseeds, tag = "A_geo", None, None
    scen = rats = tsds = None
    ssy = False
    hmiss, hpat, hseed = 0.0, "point", 0
    rest = []
    for a in argv:
        if a.startswith("--graph="):
            graph = a.split("=", 1)[1]
        elif a.startswith("--mask-seeds="):      # e.g. --mask-seeds=0-9 or 0,1,2
            v = a.split("=", 1)[1]
            mseeds = ([int(x) for x in v.split(",")] if "," in v else
                      list(range(int(v.split("-")[0]), int(v.split("-")[1]) + 1))
                      if "-" in v else [int(v)])
        elif a.startswith("--tag="):
            tag = a.split("=", 1)[1]
        elif a.startswith("--scenarios="):
            scen = a.split("=", 1)[1].split(",")
        elif a.startswith("--ratios="):
            rats = [float(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--train-seeds="):
            tsds = [int(x) for x in a.split("=", 1)[1].split(",")]
        elif a == "--split-shock-years":
            ssy = True
        elif a.startswith("--hist-missing="):
            hmiss = float(a.split("=", 1)[1])
        elif a.startswith("--hist-pattern="):
            hpat = a.split("=", 1)[1]
        elif a.startswith("--hist-seed="):
            hseed = int(a.split("=", 1)[1])
        else:
            rest.append(a)
    main(rest or ALL_MODELS, conf, graph, mseeds, tag, scen, rats, tsds, ssy,
         hmiss, hpat, hseed)
