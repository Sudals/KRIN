"""Phase 5 — Training loop shared by all neural models.

Masking policy:
  TRAIN  a fresh random spatial mask is sampled per sample/batch (ratio drawn
         from the configured set) — inductive style, prevents overfitting to a
         single hidden subset.
  VAL    fixed mask from masks.json for the requested (ratio, strategy, seed),
         so model selection matches the evaluation protocol.

Loss is masked MSE on hidden & valid nodes in scaled space. Early stopping
tracks val MAE in ORIGINAL scale on hidden nodes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
from kriging_data import load_bundle, make_dataset
from models import build_model


def _sample_masks(B, N, ratios, device, rng):
    """One random observed mask per sample in the batch."""
    obs = torch.ones(B, N, device=device)
    for b in range(B):
        r = float(ratios[rng.integers(len(ratios))])
        hide = rng.choice(N, size=int(round(r * N)), replace=False)
        obs[b, hide] = 0.0
    return obs


def masked_mse(pred, target, hidden, valid):
    w = hidden * valid
    denom = w.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * w).sum() / denom


@torch.no_grad()
def evaluate(model, loader, bundle, fixed_obs, device):
    model.eval()
    scaler = bundle.scaler
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.std_, dtype=torch.float32, device=device)
    obs = torch.tensor(fixed_obs, dtype=torch.float32, device=device)
    preds, gts, masks = [], [], []
    for hist, x_t, y_raw, valid in loader:
        hist, x_t, valid = hist.to(device), x_t.to(device), valid.to(device)
        # A node with no ground truth on day t cannot be an *observed* input:
        # the gap was filled with 0 in scaled space, which means "this node's
        # pre-shock mean level", not "missing". Feeding that as an observation
        # injects a fictitious value (see LTFJ/LTBA, 2020-05-24).
        ob = obs.expand(hist.size(0), -1) * valid
        x_obs = x_t * ob
        out = model(hist, x_obs, ob)                          # scaled
        inv = torch.expm1(out * std + mean).clamp_min(0) if scaler.log1p \
            else (out * std + mean)
        hidden = (1 - ob) * valid
        preds.append(inv.cpu().numpy())
        gts.append(y_raw.numpy())
        masks.append(hidden.cpu().numpy().astype(bool))
    p, g, m = np.concatenate(preds), np.concatenate(gts), np.concatenate(masks)
    return utils.metrics(g, p, m)


def train_one(cfg, scenario, model_name, ratio, strategy, seed,
              graph_override=None, verbose=True, train_frac=1.0, train_seed=None):
    """Train one model.

    ``seed`` selects the evaluation MASK; ``train_seed`` seeds the training
    randomness (weight init, batch shuffling, dynamic masks). They are separate
    knobs: leaving ``train_seed`` at None reproduces the historical behaviour of
    every run sharing ``cfg["seed"]``, so the only spread in the old results was
    across mask draws, not across training runs.
    """
    ts = cfg["seed"] if train_seed is None else int(train_seed)
    utils.set_seed(ts)
    device = utils.get_device(cfg["train"]["device"])
    rng = np.random.default_rng(ts)

    bundle = load_bundle(cfg, scenario, ratio, strategy, seed, graph_override)
    N, K = len(bundle.nodes), bundle.K
    train_idx = bundle.train
    if train_frac < 1.0:                       # data-efficiency experiments
        keep = rng.choice(len(train_idx), size=int(len(train_idx) * train_frac),
                          replace=False)
        train_idx = [train_idx[i] for i in sorted(keep)]
    model = build_model(model_name, N, K, cfg).to(device)
    # The faithful-reimplementation arm runs each backbone at the batch size and
    # learning rate its own paper used, so an adapter carries them as HP and
    # overrides the shared defaults here. Everything else stays common.
    hp = getattr(type(getattr(model, "inner", model)), "HP", {})
    bs = hp.get("batch", cfg["train"]["batch_size"])
    tl = DataLoader(make_dataset(bundle, train_idx),
                    batch_size=bs, shuffle=True, drop_last=True)
    vl = DataLoader(make_dataset(bundle, bundle.val), batch_size=bs)

    model.set_graph(torch.tensor(bundle.A_hat, dtype=torch.float32, device=device),
                    torch.tensor(bundle.A_raw, dtype=torch.float32, device=device))
    # Train-derived level prior. Handed identically to our model and to any wrapped
    # baseline, so neither side gets a centring the other does not.
    if hasattr(model, "set_level_prior"):
        model.set_level_prior(bundle.dow_weight)
    if hasattr(model, "set_support_stats") and bundle.rho is not None:
        model.set_support_stats(
            *(torch.tensor(v, dtype=torch.float32, device=device)
              for v in (bundle.rho, bundle.q_lo, bundle.q_hi)))
    if hasattr(model, "set_tv_graphs") and bundle.components is not None:
        c = {k: torch.tensor(v, dtype=torch.float32, device=device)
             for k, v in bundle.components.items()}
        model.set_tv_graphs(c["geo_hat"], c["corr_hat"], c["geo_raw"], c["corr_raw"])
    if hasattr(model, "set_mr_graphs"):
        from pathlib import Path as _P
        proc = _P(cfg["paths"]["processed"])
        pcorr = "A_pcorr_hat.npy" if scenario == "inregime" else "A_pcorr_pre_hat.npy"
        geo_hat = torch.tensor(np.load(proc / "A_geo_hat.npy"), dtype=torch.float32, device=device)
        pcorr_hat = torch.tensor(np.load(proc / pcorr), dtype=torch.float32, device=device)
        geo_raw = torch.tensor(np.load(proc / "A_geo.npy"), dtype=torch.float32, device=device)
        model.set_mr_graphs([geo_hat, pcorr_hat], geo_raw)
    opt = torch.optim.Adam(model.parameters(), lr=hp.get("lr", cfg["train"]["lr"]),
                           weight_decay=cfg["train"]["weight_decay"])

    mean = torch.tensor(bundle.scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(bundle.scaler.std_, dtype=torch.float32, device=device)
    ratios = cfg["mask"]["ratios"]

    best_mae, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tot = 0.0
        for hist, x_t, y_raw, valid in tl:
            hist, x_t, valid = hist.to(device), x_t.to(device), valid.to(device)
            ob = _sample_masks(hist.size(0), N, ratios, device, rng) * valid
            x_obs = x_t * ob
            out = model(hist, x_obs, ob)
            loss = masked_mse(out, x_t, hidden=(1 - ob), valid=valid)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        val = evaluate(model, vl, bundle, bundle.observed, device)
        history.append({"epoch": epoch, "train_loss": tot / len(tl), **val})
        if verbose and (epoch % 5 == 0 or epoch == cfg["train"]["epochs"] - 1):
            print(f"  e{epoch:02d} loss={tot/len(tl):.4f} valMAE={val['MAE']:.2f} "
                  f"valR2={val['R2']:.3f}")
        if val["MAE"] < best_mae - 1e-4:
            best_mae, best_state, bad = val["MAE"], \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                if verbose:
                    print(f"  early stop @ e{epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, bundle, history, best_mae


def main(config, scenario, model_name, ratio, strategy, seed):
    cfg = utils.load_config(config)
    print(f"[train] {model_name} | {scenario} r={ratio} {strategy} s{seed}")
    model, bundle, history, best = train_one(
        cfg, scenario, model_name, ratio, strategy, seed)
    ckpt_dir = Path(cfg["paths"]["results"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}_{scenario}_r{ratio}_{strategy}_s{seed}"
    torch.save({"state_dict": model.state_dict(), "history": history}, ckpt_dir / f"{tag}.pt")
    print(f"[done] best valMAE={best:.2f} → {ckpt_dir/(tag+'.pt')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--scenario", default="inregime")
    ap.add_argument("--model", default="stgnn")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--strategy", default="random")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.config, a.scenario, a.model, a.ratio, a.strategy, a.seed)
