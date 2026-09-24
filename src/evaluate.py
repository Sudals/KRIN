"""Phase 6 — Experiments, ablations, and plots.

Protocol (token/compute-efficient and leakage-safe):
  * Train each neural model ONCE per (model, scenario, ratio) with dynamic
    random masks, selecting on the val mask (random, seed 0).
  * Evaluate the trained model on the TEST set against the FIXED masks for
    strategies × seeds, broken down by regime → mean ± std over seeds.

Outputs:
  results/main_results.csv     model × scenario × ratio × regime (mean±std)
  results/ablation_graph.csv   STGNN with geo / corr / mixed graphs
  results/figures/*.png        comparison plots + training curves
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
from kriging_data import load_bundle, make_dataset
from train import train_one

REGIMES_ORDER = ["all", "pre", "shock", "recovery"]


@torch.no_grad()
def test_metrics(model, bundle, fixed_obs, regime_arr, device):
    """Per-regime metrics on the test set with a fixed observed mask."""
    model.eval()
    scaler = bundle.scaler
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.std_, dtype=torch.float32, device=device)
    obs = torch.tensor(fixed_obs, dtype=torch.float32, device=device)
    # Official SPIN's attention is quadratic in the window, so it cannot take the
    # usual evaluation batch on the larger graphs. An adapter that declares its
    # paper's batch size is evaluated at no more than that.
    inner = getattr(model, "inner", model)          # KRIN wraps the backbone
    bs = min(64, getattr(type(inner), "HP", {}).get("batch", 64))
    loader = DataLoader(make_dataset(bundle, bundle.test), batch_size=bs)

    preds, gts, masks = [], [], []
    for hist, x_t, y_raw, valid in loader:
        hist, x_t, valid = hist.to(device), x_t.to(device), valid.to(device)
        # Never present a NaN-filled cell as observed — see train.evaluate().
        ob = obs.expand(hist.size(0), -1) * valid
        out = model(hist, x_t * ob, ob)
        inv = torch.expm1(out * std + mean).clamp_min(0) if scaler.log1p \
            else (out * std + mean)
        preds.append(inv.cpu().numpy())
        gts.append(y_raw.numpy())
        masks.append(((1 - ob) * valid).cpu().numpy().astype(bool))
    p, g, m = np.concatenate(preds), np.concatenate(gts), np.concatenate(masks)

    regs = regime_arr[bundle.test]
    rows = {"all": utils.metrics(g, p, m)}
    for reg in pd.unique(regs):
        sel = regs == reg
        rows[reg] = utils.metrics(g[sel], p[sel], m[sel])
    return rows


def aggregate_over_seeds(per_seed: list[dict]) -> dict:
    """per_seed: list of {regime: metric_dict}. Returns regime → mean/std for
    every metric key present (except the sample count 'n')."""
    out = {}
    regimes = list(per_seed[0].keys())
    keys = [k for k in per_seed[0][regimes[0]] if k != "n"]
    for reg in regimes:
        agg = {}
        for k in keys:
            vals = [d[reg][k] for d in per_seed if not np.isnan(d[reg].get(k, np.nan))]
            agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
        out[reg] = agg
    return out


def run_main(cfg, models, scenarios, ratios, device, regime_arr):
    records = []
    for model_name, scenario, ratio in itertools.product(models, scenarios, ratios):
        print(f"\n=== TRAIN {model_name} | {scenario} | r={ratio} ===")
        model, bundle, history, best = train_one(
            cfg, scenario, model_name, ratio, "random", 0, verbose=True)
        _save_curve(history, cfg, f"{model_name}_{scenario}_r{ratio}")
        for strategy in cfg["mask"]["strategies"]:
            per_seed = []
            for seed in cfg["mask"]["seeds"]:
                b = load_bundle(cfg, scenario, ratio, strategy, seed,
                                graph_override=bundle_graph(cfg, scenario))
                per_seed.append(test_metrics(model, b, b.observed, regime_arr, device))
            agg = aggregate_over_seeds(per_seed)
            for reg, m in agg.items():
                records.append({"model": model_name, "scenario": scenario,
                                "ratio": ratio, "strategy": strategy,
                                "regime": reg, **m})
            print(f"  [{strategy}] all MAE={agg['all']['MAE_mean']:.2f}"
                  f"±{agg['all']['MAE_std']:.2f} R2={agg['all']['R2_mean']:.3f}")
    return pd.DataFrame(records)


def bundle_graph(cfg, scenario):
    return utils.load_json(Path(cfg["paths"]["processed"]) /
                           "splits.json")["scenarios"][scenario]["graph"]


def run_graph_ablation(cfg, device, regime_arr):
    """STGNN on inregime, ratio 0.5, with geo / corr / mixed graphs."""
    records = []
    for graph in ("A_geo", "A_corr", "A_mixed"):
        print(f"\n=== ABLATION STGNN | graph={graph} ===")
        model, bundle, _, _ = train_one(cfg, "inregime", "stgnn", 0.5,
                                        "random", 0, graph_override=graph, verbose=False)
        per_seed = []
        for s in cfg["mask"]["seeds"]:
            b = load_bundle(cfg, "inregime", 0.5, "random", s, graph_override=graph)
            per_seed.append(test_metrics(model, b, b.observed, regime_arr, device))
        agg = aggregate_over_seeds(per_seed)["all"]
        records.append({"graph": graph, **agg})
        print(f"  all MAE={agg['MAE_mean']:.2f}±{agg['MAE_std']:.2f} "
              f"R2={agg['R2_mean']:.3f}")
    return pd.DataFrame(records)


def _save_curve(history, cfg, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(h.epoch, h.train_loss, "b-", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(h.epoch, h.MAE, "r-", label="val MAE")
    ax2.set_ylabel("val MAE", color="r")
    fig.suptitle(tag); fig.tight_layout()
    d = Path(cfg["paths"]["figures"]) / "curves"; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{tag}.png", dpi=110); plt.close(fig)


def _plot_main(df, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = df[(df.regime == "all") & (df.strategy == "random")]
    scenarios = sorted(res.scenario.unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4.2), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, sc in zip(axes, scenarios):
        sub = res[res.scenario == sc]
        for model in sorted(sub.model.unique()):
            s = sub[sub.model == model].sort_values("ratio")
            ax.errorbar(s.ratio, s.MAE_mean, yerr=s.MAE_std, marker="o", capsize=3, label=model)
        ax.set_title(sc); ax.set_xlabel("mask ratio r"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("MAE (hidden nodes, original scale)")
    axes[-1].legend()
    fig.suptitle("Reconstruction error vs. mask ratio")
    fig.tight_layout()
    fig.savefig(Path(cfg["paths"]["figures"]) / "main_mae_vs_ratio.png", dpi=120)
    plt.close(fig)


def main(config):
    cfg = utils.load_config(config)
    device = utils.get_device(cfg["train"]["device"])
    proc = Path(cfg["paths"]["processed"])
    regime_arr = np.array(utils.load_json(proc / "dates.json")["regime"])
    res = Path(cfg["paths"]["results"]); res.mkdir(parents=True, exist_ok=True)

    # neural models; HA/KNN baselines live in baselines.csv
    models = ["gru", "ignnk", "satcn", "grin", "spin", "kits", "stgnn", "stgnnr"]
    scenarios = list(utils.load_json(proc / "splits.json")["scenarios"])
    ratios = cfg["mask"]["ratios"]

    main_df = run_main(cfg, models, scenarios, ratios, device, regime_arr)
    main_df.to_csv(res / "main_results.csv", index=False)
    _plot_main(main_df, cfg)

    abl = run_graph_ablation(cfg, device, regime_arr)
    abl.to_csv(res / "ablation_graph.csv", index=False)

    print("\n=== MAIN (regime=all, random, mean over seeds) ===")
    piv = main_df[(main_df.regime == "all") & (main_df.strategy == "random")].pivot_table(
        index=["scenario", "ratio"], columns="model", values="MAE_mean").round(2)
    print(piv.to_string())
    print(f"\n[done] results written to {res}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
