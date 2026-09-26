"""Re-run coverage models on the exact SNaive-7 scoring support.

Writes coverage_matched_runs.csv, checkpoints, and per-target predictions.
New runs go in a separate directory; released CSVs remain unchanged.
Training and input masks are
identical to run_coverage.py; only scoring additionally requires a valid lag-7
target value. MAE_all_valid records the old scoring rule as a replay check.

The original executed file is archived under experiments/provenance/.
See analysis/MATCHED_RERUN.md for the recorded configuration and environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import kriging_data as kd
import utils
from train import train_one
from models import build_model
from run_coverage import placement, TRAIN_SEEDS, PLACEMENT_SEEDS, SCEN, RATIO
from run_history import free_on, region, realised_shift, REGION_FRAC, ALPHA


def matched_score_mask(valid, score, lag_valid):
    return (valid > 0) & score[None, :] & lag_valid


@torch.no_grad()
def evaluate_matched(model, bundle, obs, score, device, max_batch_size=16):
    model.eval()
    scaler = bundle.scaler
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.std_, dtype=torch.float32, device=device)
    observed = torch.tensor(obs, dtype=torch.float32, device=device)
    targets = torch.tensor(score, dtype=torch.bool, device=device)
    inner = getattr(model, "inner", model)
    batch_size = min(max_batch_size, getattr(type(inner), "HP", {}).get("batch", 64))
    loader = DataLoader(kd.make_dataset(bundle, bundle.test), batch_size=batch_size)
    test = np.asarray(bundle.test)
    lag_valid = ~np.isnan(bundle.raw[test - 7])
    error, error_all, n, n_all, offset = 0.0, 0.0, 0, 0, 0
    predictions = []
    for hist, x_t, y_raw, valid in loader:
        hist, x_t, valid = hist.to(device), x_t.to(device), valid.to(device)
        ob = observed.expand(hist.size(0), -1) * valid
        out = model(hist, x_t * ob, ob)
        pred = (torch.expm1(out * std + mean).clamp_min(0) if scaler.log1p
                else out * std + mean)
        mask_all = (valid > 0) & targets[None, :]
        lag = torch.tensor(lag_valid[offset:offset + len(hist)], device=device)
        mask = matched_score_mask(valid, targets, lag)
        absolute_error = (pred - y_raw.to(device)).abs()
        error += float(absolute_error[mask].sum())
        error_all += float(absolute_error[mask_all].sum())
        n += int(mask.sum())
        n_all += int(mask_all.sum())
        predictions.append(pred[:, targets].cpu().numpy())
        offset += len(hist)
    if not n:
        raise ValueError("No common target/lag-7 scoring support")
    payload = {"prediction": np.concatenate(predictions),
               "truth": bundle.raw[np.ix_(test, np.flatnonzero(score))],
               "lag7": bundle.raw[np.ix_(test - 7, np.flatnonzero(score))],
               "target_indices": np.flatnonzero(score), "test_indices": test}
    return error / n, error_all / n_all, n, n_all, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--models", default="spin,spin_c,ignnk_off,ignnk_off_c")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <config paths.results>/coverage_matched_rerun")
    parser.add_argument("--resume", action="store_true", help="Reuse this rerun's saved checkpoints and complete rows")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = utils.load_config(args.config)
    device = utils.get_device(cfg["train"]["device"])
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this run; no automatic CPU fallback")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, torch.cuda.current_device())
    name = Path(args.config).stem
    source = Path(cfg["paths"]["results"]) / "coverage_runs.csv"
    res = args.output_dir or source.parent / "coverage_matched_rerun"
    original = pd.read_csv(source).set_index(["model", "train_seed", "placement_seed", "shift", "coverage"])
    output = res / "coverage_matched_runs.csv"
    if output.exists() and not args.resume:
        raise FileExistsError(f"Preserving existing rerun: {output}")
    pred_dir = res / "coverage_matched_predictions"
    checkpoint_dir = res / "coverage_matched_checkpoints"
    pred_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "config": args.config, "torch": torch.__version__, "numpy": np.__version__,
                "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                "device": str(device), "score_policy": "target_and_lag7_valid",
                "models": args.models.split(","), "train_seeds": TRAIN_SEEDS,
                "placement_seeds": PLACEMENT_SEEDS, "scenario": SCEN,
                "training_hidden_ratio": RATIO, "regional_alpha": ALPHA,
                "region_fraction": REGION_FRAC, "threads": args.threads,
                "cuda_memory_fraction": args.cuda_memory_fraction,
                "evaluation_batch_size_limit": args.eval_batch_size,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "experiment_code_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__), Path(__file__).with_name("run_coverage.py"),
                              Path(__file__).with_name("run_history.py"))},
                "source_code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((ROOT / "src").rglob("*.py"))}}
    provenance = res / "coverage_matched_provenance.json"
    if args.resume and provenance.exists():
        previous_metadata = json.loads(provenance.read_text())
        for field in ("source_sha256", "source_code_sha256", "config", "config_sha256",
                      "experiment_code_sha256", "evaluation_batch_size_limit", "threads"):
            if metadata[field] != previous_metadata.get(field):
                raise ValueError(f"Cannot resume after {field} changed")
        metadata["previous_runner_sha256"] = previous_metadata["script_sha256"]
    provenance.write_text(json.dumps(metadata, indent=2) + "\n")
    N = np.load(Path(cfg["paths"]["processed"]) / "A_geo.npy").shape[0]
    moved = region(cfg, REGION_FRAC)
    magnitude = realised_shift(cfg, ALPHA, moved)
    plans = {p: placement(cfg, moved, N, p) for p in PLACEMENT_SEEDS}
    rows = pd.read_csv(output).to_dict("records") if args.resume and output.exists() else []
    started = time.monotonic()
    print(f"{name}: matched scoring on {device}; {args.models}", flush=True)
    for model_name in args.models.split(","):
        for train_seed in TRAIN_SEEDS:
            completed = [r for r in rows if r["model"] == model_name and r["train_seed"] == train_seed]
            if len(completed) == 20:
                continue
            if completed:
                raise ValueError("Unexpected partial seed in saved rows")
            kd.set_level_shift(1.0, "pre")
            kd.set_history_staleness(0)
            checkpoint = checkpoint_dir / f"{model_name}_ts{train_seed}.pt"
            if args.resume and checkpoint.exists():
                print(f"Restoring {model_name} seed {train_seed}", flush=True)
                bundle = kd.load_bundle(cfg, SCEN, RATIO, "random", 0, graph_override="A_geo")
                model = build_model(model_name, N, bundle.K, cfg).to(device)
                model.set_graph(torch.tensor(bundle.A_hat, dtype=torch.float32, device=device),
                                torch.tensor(bundle.A_raw, dtype=torch.float32, device=device))
                if hasattr(model, "set_level_prior"):
                    model.set_level_prior(bundle.dow_weight)
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                if saved["model"] != model_name or saved["train_seed"] != train_seed:
                    raise ValueError("Checkpoint model/seed mismatch")
                model.load_state_dict(saved["state_dict"])
            else:
                print(f"Training {model_name} seed {train_seed}", flush=True)
                model, _, history, best = train_one(cfg, SCEN, model_name, RATIO, "random", 0,
                                                   graph_override="A_geo", verbose=False,
                                                   train_seed=train_seed)
                torch.save({"state_dict": model.state_dict(), "history": history,
                            "best_validation_mae": best, "model": model_name, "train_seed": train_seed},
                           checkpoint)
            for shift, alpha in (("no shift", 1.0), ("regional shift", ALPHA)):
                kd.set_level_shift(alpha, "pre", moved if alpha != 1.0 else None)
                bundle = kd.load_bundle(cfg, SCEN, RATIO, "random", 0, graph_override="A_geo")
                for p, (score, obs, n_obs) in plans.items():
                    for coverage in ("inside", "outside"):
                        mae, mae_all, n, n_all, payload = evaluate_matched(
                            model, bundle, obs[coverage], score, device, args.eval_batch_size)
                        sn, cm = free_on(bundle, obs[coverage], score)
                        if not np.isfinite([mae, mae_all, sn, cm]).all():
                            raise ValueError("Nonfinite evaluation result")
                        previous = original.loc[(model_name, train_seed, p, shift, coverage)]
                        if not np.allclose([sn, cm], previous[["snaive7", "common_mode"]].to_numpy(dtype=float)):
                            raise ValueError("Baseline no longer matches original run")
                        tag = f"{model_name}_ts{train_seed}_p{p}_{shift.replace(' ', '_')}_{coverage}"
                        prediction_path = pred_dir / f"{tag}.npz"
                        np.savez_compressed(prediction_path, **payload)
                        rows.append({"dataset": name, "model": model_name, "train_seed": train_seed,
                                     "placement_seed": p, "shift": shift, "coverage": coverage,
                                     "MAE": mae, "snaive7": sn, "common_mode": cm,
                                     "n_target": int(score.sum()), "n_observed": n_obs,
                                     "realised_shift": magnitude, "model_score_policy": "target_and_lag7_valid",
                                     "n_scored": n, "n_all_valid": n_all, "MAE_all_valid": mae_all,
                                     "previous_MAE": float(previous.MAE),
                                     "replay_MAE_difference": mae_all - float(previous.MAE),
                                     "prediction_file": str(prediction_path),
                                     "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
            pd.DataFrame(rows).to_csv(output, index=False, float_format="%.17g")
            print(f"Finished {model_name} seed {train_seed}; {len(rows)} rows; "
                  f"{(time.monotonic() - started) / 60:.1f} min", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    kd.set_level_shift(1.0, "pre")
    print(f"Done: {output}", flush=True)


if __name__ == "__main__":
    main()
