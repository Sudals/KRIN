"""Recalculate paired coverage contrasts with matched SNaive-7 denominators.

The 3 training seeds and 5 placements are crossed factors, not 15 independent
runs. Form all contrasts before independently resampling the two factors.
Optionally reconstruct the training-free errors and audit scoring support from
the processed data, without training any model or changing the original CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {"cta": "Chicago", "mta": "Subway"}
MODELS = ("spin", "spin_c", "ignnk_off", "ignnk_off_c")
KEY = ["train_seed", "placement_seed"]
CONDITION = ["placement_seed", "shift", "coverage"]
FULL_KEY = ["model", *KEY, "shift", "coverage"]
SHIFTS = ("no shift", "regional shift")
COVERAGES = ("inside", "outside")
METRICS = ("MAE", "MAE_over_snaive7")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def validate_runs(d, dataset):
    required = set(FULL_KEY + ["dataset", "MAE", "snaive7", "common_mode",
                               "n_target", "n_observed", "realised_shift"])
    if required - set(d):
        raise ValueError(f"{dataset}: missing columns {sorted(required - set(d))}")
    if d.empty or d[list(required)].isna().any().any():
        raise ValueError(f"{dataset}: empty input or missing required values")
    if set(d.dataset) != {dataset}:
        raise ValueError(f"{dataset}: unexpected dataset labels")
    if d.duplicated(FULL_KEY).any():
        raise ValueError(f"{dataset}: duplicate pairing keys")
    expected = pd.MultiIndex.from_product(
        [MODELS, range(3), range(5), SHIFTS, COVERAGES], names=FULL_KEY)
    actual = pd.MultiIndex.from_frame(d[FULL_KEY])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError(f"{dataset}: incomplete or unexpected 4 x 3 x 5 x 2 x 2 grid")
    values = d[["MAE", "snaive7", "common_mode"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (d.snaive7 <= 0).any():
        raise ValueError(f"{dataset}: invalid errors or nonpositive SNaive-7 denominator")
    # Repeated reference errors differ at most by floating-point serialization.
    for col in ("snaive7", "common_mode", "n_target", "n_observed", "realised_shift"):
        ref = d.groupby(CONDITION)[col].transform("first")
        if not np.allclose(d[col], ref, rtol=1e-10, atol=1e-9):
            raise ValueError(f"{dataset}: {col} differs across model/training replicas")
    p = d.pivot(index=["model", *KEY, "shift"], columns="coverage", values="snaive7")
    if not np.allclose(p.inside, p.outside, rtol=1e-10, atol=1e-9):
        raise ValueError(f"{dataset}: SNaive-7 unexpectedly depends on coverage")
    if "model_score_policy" in d and set(d.model_score_policy) != {"target_and_lag7_valid"}:
        raise ValueError(f"{dataset}: unknown or mixed explicit scoring policies")
    return d.sort_values(FULL_KEY).reset_index(drop=True)


def verify_baselines(root, dataset, d):
    """Independently reconstruct the saved protocol using only numpy/pandas.

    This also checks a subtle condition not established by equal run keys:
    learned MAE scores any target with ground truth, whereas free_on requires
    a valid lag-7 value as well. Do not call those supports equal without checking.
    """
    proc = root / "data" / f"processed_{dataset}"
    paths = {name: proc / name for name in
             ("traffic_matrix.parquet", "splits.json", "A_geo.npy", "dates.json")}
    matrix = pd.read_parquet(paths["traffic_matrix.parquet"])
    raw = matrix.to_numpy(dtype=float)
    common_support = "model_score_policy" in d
    splits = json.loads(paths["splits.json"].read_text())
    dates = json.loads(paths["dates.json"].read_text())["dates"]
    test = np.asarray(splits["scenarios"]["inregime"]["test"], dtype=int)
    if (test < max(7, splits["window_K"])).any():
        raise ValueError("Test targets do not have a full lag/history window")
    adjacency = np.load(paths["A_geo.npy"])
    n_nodes = len(adjacency)
    region = np.sort(np.argsort(-adjacency[int(np.argmax(adjacency.sum(1)))])[
        :max(1, int(round(0.30 * n_nodes)))])
    rows, missing = [], []
    for placement_seed in range(5):
        rng = np.random.default_rng(1000 + placement_seed)
        perm = rng.permutation(region)
        score = np.zeros(n_nodes, dtype=bool)
        score[perm[:max(1, len(region) // 2)]] = True
        inside = np.array([i for i in region if not score[i]])
        outside = np.setdiff1d(np.arange(n_nodes), region)
        n_obs = min(len(inside), len(outside))
        observations = {}
        for cov, pool in zip(COVERAGES, (inside, outside)):
            obs = np.zeros(n_nodes, dtype=bool)
            obs[rng.permutation(pool)[:n_obs]] = True
            observations[cov] = obs
        gt, lag = raw[test], raw[test - 7]
        model_support = score[None, :] & ~np.isnan(gt)
        reference_support = model_support & ~np.isnan(lag)
        n_all, n_ref = int(model_support.sum()), int(reference_support.sum())
        n_model = n_ref if common_support else n_all
        if not n_ref:
            raise ValueError(f"{dataset}: empty baseline scoring support")
        for day, node in np.argwhere(model_support & ~reference_support):
            missing.append({"dataset": dataset, "placement_seed": placement_seed,
                            "date": dates[int(test[day])], "node": str(matrix.columns[node]),
                            "included_in_model_score": not common_support})
        for shift, alpha in zip(SHIFTS, (1.0, 0.25)):
            shifted = raw.copy()
            onset = max(0, int(test.min()) - splits["window_K"])
            shifted[np.ix_(np.arange(onset, len(raw)), region)] *= alpha
            for coverage, obs in observations.items():
                err_sn, err_cm, count = 0.0, 0.0, 0
                for t in test:
                    y, y7 = shifted[t], shifted[t - 7]
                    hidden = score & ~np.isnan(y) & ~np.isnan(y7)
                    if not hidden.any():
                        continue
                    seen = obs & ~np.isnan(y) & ~np.isnan(y7)
                    delta = (np.median(np.log1p(np.clip(y[seen], 0, None))
                                       - np.log1p(np.clip(y7[seen], 0, None)))
                             if seen.any() else 0.0)
                    cm = np.expm1(np.log1p(np.clip(y7[hidden], 0, None)) + delta).clip(0)
                    err_sn += float(np.abs(y[hidden] - y7[hidden]).sum())
                    err_cm += float(np.abs(y[hidden] - cm).sum())
                    count += int(hidden.sum())
                sn, cm = err_sn / count, err_cm / count
                saved = d[(d.placement_seed == placement_seed)
                          & (d["shift"] == shift) & (d.coverage == coverage)]
                for col, value in (("snaive7", sn), ("common_mode", cm),
                                   ("n_target", int(score.sum())), ("n_observed", n_obs)):
                    if not np.allclose(saved[col], value, rtol=1e-10, atol=1e-9):
                        raise ValueError(f"{dataset}: recomputed {col} disagrees at "
                                         f"{placement_seed}, {shift}, {coverage}")
                if common_support:
                    if not (saved.n_scored.eq(n_ref).all() and saved.n_all_valid.eq(n_all).all()):
                        raise ValueError(f"{dataset}: saved matched scoring counts disagree")
                rows.append({"dataset": dataset, "placement_seed": placement_seed,
                             "shift": shift, "coverage": coverage,
                             "snaive7_recomputed": sn, "common_mode_recomputed": cm,
                             "max_snaive7_abs_difference": float(np.abs(saved.snaive7 - sn).max()),
                             "max_common_mode_abs_difference": float(np.abs(saved.common_mode - cm).max()),
                             "n_model_cells": n_model, "n_snaive7_cells": n_ref,
                             "n_all_valid_target_cells": n_all,
                             "n_lag7_missing_target_cells": n_all - n_ref,
                             "n_unmatched_cells": n_model - n_ref,
                             "score_support_status": "exact" if n_model == n_ref else "mismatch",
                             "target_indices_sha256": index_hash(np.flatnonzero(score)),
                             "test_indices_sha256": index_hash(test),
                             "test_days": len(test), "test_start": dates[int(test.min())],
                             "test_end": dates[int(test.max())]})
    return pd.DataFrame(rows), missing, {str(p.relative_to(root)): sha256(p) for p in paths.values()}


def verify_predictions(root, d, references):
    """Recompute every saved rerun MAE from its predictions and common mask."""
    required = {"prediction_file", "prediction_sha256", "n_scored", "n_all_valid", "MAE_all_valid"}
    if required - set(d):
        raise ValueError("Prediction verification requires the matched rerun files")
    refs = references.set_index(CONDITION)
    max_matched, max_all = 0.0, 0.0
    for row in d.itertuples():
        path = root / row.prediction_file
        if sha256(path) != row.prediction_sha256:
            raise ValueError(f"Prediction hash mismatch: {path}")
        ref = refs.loc[(row.placement_seed, row.shift, row.coverage)]
        with np.load(path, allow_pickle=False) as payload:
            if index_hash(payload["target_indices"]) != ref.target_indices_sha256:
                raise ValueError(f"Wrong target nodes: {path}")
            if index_hash(payload["test_indices"]) != ref.test_indices_sha256:
                raise ValueError(f"Wrong target days: {path}")
            truth, lag, pred = payload["truth"], payload["lag7"], payload["prediction"]
            if not np.isfinite(pred).all() or pred.shape != truth.shape or lag.shape != truth.shape:
                raise ValueError(f"Invalid prediction array: {path}")
            all_valid = np.isfinite(truth)
            valid = all_valid & np.isfinite(lag)
            if int(valid.sum()) != row.n_scored or int(all_valid.sum()) != row.n_all_valid:
                raise ValueError(f"Prediction scoring count mismatch: {path}")
            # Match the evaluator's float32 subtraction, but sum in float64.
            err = np.abs(pred - truth.astype(np.float32))
            matched = float(err[valid].mean(dtype=np.float64))
            full = float(err[all_valid].mean(dtype=np.float64))
            if not np.allclose([matched, full], [row.MAE, row.MAE_all_valid], rtol=3e-7, atol=1e-6):
                raise ValueError(f"Prediction MAE disagrees: {path}")
            snaive = float(np.abs(truth - lag)[valid].mean())
            if not np.isclose(snaive, row.snaive7, rtol=1e-10, atol=1e-9):
                raise ValueError(f"Prediction support SNaive-7 disagrees: {path}")
            max_matched = max(max_matched, abs(matched - row.MAE))
            max_all = max(max_all, abs(full - row.MAE_all_valid))
    return {"prediction_files_verified": len(d), "max_matched_mae_abs_difference": max_matched,
            "max_all_valid_mae_abs_difference": max_all,
            "replay_max_abs_mae_change": float(d.replay_MAE_difference.abs().max()),
            "replay_mean_abs_mae_change": float(d.replay_MAE_difference.abs().mean())}


def paired_contrasts(d):
    """Return contrasts without silently intersecting keys or dropping NaNs."""
    records = []
    for metric in METRICS:
        extras = {}
        for model in MODELS:
            pivot = d[d.model == model].pivot(index=KEY, columns=["shift", "coverage"],
                                              values=metric).sort_index()
            penalties = {shift: pivot[(shift, "outside")] - pivot[(shift, "inside")]
                         for shift in SHIFTS}
            extra = penalties[SHIFTS[1]] - penalties[SHIFTS[0]]
            extras[model] = extra
            quantities = {"outside_penalty_no_shift": penalties[SHIFTS[0]],
                          "outside_penalty_regional_shift": penalties[SHIFTS[1]],
                          "shift_excess_penalty": extra}
            for contrast, values in quantities.items():
                for (train_seed, placement_seed), value in values.items():
                    records.append({"model": model, "metric": metric, "contrast": contrast,
                                    "train_seed": train_seed, "placement_seed": placement_seed,
                                    "value": value})
        for model in ("spin", "ignnk_off"):
            reduction = extras[model] - extras[model + "_c"]
            for (train_seed, placement_seed), value in reduction.items():
                records.append({"model": model, "metric": metric, "contrast": "krin_reduction",
                                "train_seed": train_seed, "placement_seed": placement_seed,
                                "value": value})
    return pd.DataFrame(records)


def bootstrap_draws(matrix, train_draws, placement_draws):
    """One shared placement draw across all sampled training rows per resample."""
    return matrix[train_draws[:, :, None], placement_draws[:, None, :]].mean(axis=(1, 2))


def summarise(contrasts, resamples, seed):
    rng = np.random.default_rng(seed)
    # Use identical bootstrap draws for every paired model/condition contrast.
    train_draws = rng.integers(0, 3, size=(resamples, 3))
    placement_draws = rng.integers(0, 5, size=(resamples, 5))
    records = []
    for (model, metric, contrast), g in contrasts.groupby(["model", "metric", "contrast"], sort=False):
        matrix = g.pivot(index="train_seed", columns="placement_seed", values="value").to_numpy()
        if matrix.shape != (3, 5) or not np.isfinite(matrix).all():
            raise ValueError("Contrasts must retain the complete 3 x 5 pairing grid")
        samples = bootstrap_draws(matrix, train_draws, placement_draws)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        records.append({"model": model, "metric": metric, "contrast": contrast,
                        "estimate": matrix.mean(), "ci_low": lo, "ci_high": hi,
                        "positive": int((matrix > 0).sum()), "negative": int((matrix < 0).sum()),
                        "zero": int((matrix == 0).sum()), "n_pairs": matrix.size,
                        "n_train_seeds": 3, "n_placement_seeds": 5,
                        "bootstrap": "crossed_train_placement", "resamples": resamples,
                        "bootstrap_seed": seed})
    return pd.DataFrame(records)


def publication_tables(runs, contrasts, summary, input_source):
    """Table-4 D/R and the requested five-placement mean for each fitted seed.

    D is the difference of SNaive-normalised outside penalties across shifts.
    R is the paired reduction in D after KRIN. Raw-MAE excesses have separate
    names and must not be substituted for these dimensionless quantities.
    """
    lookup = summary.set_index(["dataset", "model", "metric", "contrast"])
    per_seed = contrasts.groupby(
        ["dataset", "model", "metric", "contrast", "train_seed"]
    ).value.agg(["mean", "count"])
    table, seeds = [], []
    for dataset in DATASETS:
        for model, backbone in (("spin", "SPIN-s"), ("ignnk_off", "IGNNK-R")):
            definitions = (("D_uncentred", model, "shift_excess_penalty"),
                           ("D_KRIN", model + "_c", "shift_excess_penalty"),
                           ("R", model, "krin_reduction"))
            row = {"dataset": DATASETS[dataset], "backbone": backbone,
                   "input_source": input_source, "n_train_seeds": 3, "n_placements": 5}
            for label, variant, contrast in definitions:
                stat = lookup.loc[(dataset, variant, "MAE_over_snaive7", contrast)]
                for field in ("estimate", "ci_low", "ci_high", "positive"):
                    row[label if field == "estimate" else f"{label}_{field}"] = stat[field]
                row["score_support_status"] = stat.score_support_status
            table.append(row)
            for train_seed in range(3):
                seed_row = {"dataset": DATASETS[dataset], "backbone": backbone,
                            "train_seed": train_seed, "n_placements": 5,
                            "input_source": input_source}
                for label, variant, contrast in definitions:
                    for metric, output_label in (("MAE_over_snaive7", label),
                                                  ("MAE", "raw_" + label)):
                        stat = per_seed.loc[(dataset, variant, metric, contrast, train_seed)]
                        if stat["count"] != 5:
                            raise ValueError("A training-seed mean must contain exactly five placements")
                        seed_row[output_label] = stat["mean"]
                seeds.append(seed_row)
    mae = runs.groupby(["dataset", "model", "train_seed", "shift", "coverage"], as_index=False).agg(
        MAE=("MAE", "mean"), snaive7=("snaive7", "mean"),
        MAE_over_snaive7=("MAE_over_snaive7", "mean"), n_placements=("placement_seed", "nunique"))
    mae["dataset"] = mae.dataset.map(DATASETS)
    mae["input_source"] = input_source
    return {"table4.csv": pd.DataFrame(table),
            "training_seed_means.csv": pd.DataFrame(seeds),
            "training_seed_mae.csv": mae}


def training_seed_report(seeds, input_source):
    lines = ["# Coverage results by training seed", "",
             f"Input source: **{input_source}**. Each entry averages placement seeds 0–4",
             "within the indicated training seed. D and R are dimensionless; raw-MAE",
             "contrasts are included separately in the CSV. D_KRIN is the residual excess",
             "after KRIN, and R = D_uncentred - D_KRIN.", "",
             "| Dataset | Backbone | Training seed | D, uncentred | D, KRIN | R |",
             "|---|---|---:|---:|---:|---:|"]
    for row in seeds.itertuples():
        lines.append(f"| {row.dataset} | {row.backbone} | {row.train_seed} | "
                     f"{row.D_uncentred:.9f} | {row.D_KRIN:.9f} | {row.R:.9f} |")
    lines += ["", "`training_seed_mae.csv` also gives each fitted model's original-scale",
              "MAE averaged over five placements, separately for each shift and coverage",
              "condition. The mean of per-run ratios is retained; it is not replaced by a",
              "ratio of mean errors. Uncertainty over the crossed factors is in `table4.csv`.", ""]
    return "\n".join(lines)


def report(summary, references, audit, resamples, seed):
    lines = ["# Coverage statistics recalculation", "",
             "The original run files are unchanged. These statistics are recomputed from the",
             "full-precision rows, with a SNaive-7 denominator from the same placement and condition.",
             "", "## Matching and scoring audit", ""]
    for dataset, info in audit["datasets"].items():
        lines.append(f"- {DATASETS[dataset]}: {info['rows']} rows; unique complete 4 models x 3 training "
                     "seeds x 5 placements x 2 shifts x 2 coverage conditions.")
        r = references[references.dataset == dataset].drop_duplicates("placement_seed")
        if "n_unmatched_cells" in r and r.n_unmatched_cells.notna().all():
            total, gap = int(r.n_all_valid_target_cells.sum()), int(r.n_lag7_missing_target_cells.sum())
            unmatched = int(r.n_unmatched_cells.sum())
            lines.append(f"  Raw-data verification: {gap}/{total} target-valid entries across the five "
                         "placements have a missing lag-7 value. "
                         + ("They are excluded from both errors. " if gap and not unmatched else "")
                         + ("Model and reference supports match exactly." if not gap else
                            "Model and reference supports match exactly." if not unmatched else
                            "**The saved model and baseline errors do not have identical scoring "
                            "support. Normalised results for this dataset are provisional.**"))
        else:
            lines.append("  Scoring support is unverified; rerun with --verify-baselines and processed data.")
        if "prediction_files_verified" in info:
            lines.append(f"  Independently checked {info['prediction_files_verified']} prediction files. "
                         f"Replayed all-target MAE versus the original runs: maximum absolute change "
                         f"{info['replay_max_abs_mae_change']:.6g}. This dataset uses freshly trained "
                         "models with the original seeds and protocol; it is not a reconstruction of "
                         "unavailable original per-cell errors.")
    lines += ["", "The same date/node may occur in multiple placements; the counts above are placement",
              "scoring entries, not unique observations. Reference errors are repeated across training",
              "seeds and models; matched_snaive7.csv retains only one row per placement/shift/coverage.",
              "", "## Estimands and uncertainty", "", "```text",
              "E[m,t,p,s,c] = MAE[m,t,p,s,c] / SNaive7[p,s,c]",
              "P[m,t,p,s]   = E[m,t,p,s,outside] - E[m,t,p,s,inside]",
              "D[m,t,p]     = P[m,t,p,regional shift] - P[m,t,p,no shift]",
              "K[m,t,p]     = D[m,t,p] - D[m+KRIN,t,p]", "```", "",
              "Divide within each run before pairing or averaging. These are dimensionless ratios,",
              "not percentages, nodewise NMAE, or ratios of aggregate means. Raw MAE contrasts are",
              "also retained in summary.csv for comparison.", "",
              f"95% percentile intervals use {resamples:,} crossed bootstrap draws (RNG seed {seed}).",
              "Independently sample 3 training-seed labels and 5 placement labels with replacement,",
              "then take their Cartesian product. A placement draw is shared across all sampled",
              "training seeds; all models and conditions retain their pairing. This follows the",
              "resampling construction in [Owen (2007), The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111).",
              "With just 3 training seeds and 5 placements, these are exploratory uncertainty intervals;",
              "nominal coverage is not guaranteed. Sign counts describe 15 crossed evaluations, not",
              "15 independent replications. No p-values or equivalence claims are inferred.", "",
              "## Normalised paired contrasts", "",
              "| Dataset | Model | Contrast | Mean | 95% interval | Positive / pairs | Support |",
              "|---|---|---|---:|---|---:|---|"]
    for row in summary[summary.metric == "MAE_over_snaive7"].itertuples():
        lines.append(f"| {DATASETS[row.dataset]} | {row.model} | {row.contrast} | "
                     f"{row.estimate:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | "
                     f"{row.positive}/{row.n_pairs} | {row.score_support_status} |")
    lines += ["", "## Interpretation limits", "",
              "The regional intervention multiplies target demand by 0.25. For these saved runs,",
              "SNaive-7 MAE also scales by 0.25. A raw MAE outside penalty can therefore become",
              "smaller mechanically. The raw difference-in-differences and reductions above 100%",
              "in the previous analysis do not by themselves establish removal of a scale-adjusted",
              "coverage penalty. Use the normalised contrasts and their uncertainty instead.", "",
              "An interval containing zero does not prove equivalence or complete removal. These",
              "results are conditional on the fixed time window, regional definition and alpha=0.25.",
              "The crossed bootstrap does not resample days or regions. Exact support correction",
              "for any mismatched dataset requires model predictions on the common valid target/lag-7",
              "cells (or a rerun if predictions/checkpoints were not saved); aggregate MAEs cannot",
              "recover the missing-cell model errors.", "", "## Files", "",
              "- `normalised_runs.csv`: all input rows plus per-run ratios and support status.",
              "- `matched_snaive7.csv`: deduplicated references with audit counts, if verified.",
              "- `paired_contrasts.csv`: every per-(training seed, placement) contrast.",
              "- `summary.csv`: means, crossed bootstrap intervals, and sign counts.",
              "- `seed_means.csv`: per-training-seed and per-placement means for sensitivity inspection.",
              "- `lag7_missing_target_cells.csv`: placement/date/node entries omitted by SNaive-7,",
              "  including whether each was included in the learned-model scoring rule.",
              "- `audit.json`: input hashes, validation checks, and resampling settings.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: <root>/analysis/coverage_recalculation_<input-source>")
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-source", choices=("original", "matched"),
                        help="original: both coverage_runs.csv files; matched: use the Subway matched rerun")
    parser.add_argument("--verify-baselines", action="store_true")
    parser.add_argument("--verify-predictions", action="store_true",
                        help="Verify any input with an explicit matched scoring policy from saved predictions")
    for dataset in DATASETS:
        parser.add_argument(f"--{dataset}-runs", type=Path, help="CSV path relative to --root (or absolute)")
    args = parser.parse_args(argv)
    has_overrides = any(getattr(args, f"{ds}_runs") for ds in DATASETS)
    if args.input_source is not None and has_overrides:
        parser.error("Use --input-source or explicit run-file overrides, not both")
    if args.input_source is None and not has_overrides:
        parser.error("Choose --input-source original or matched, or supply explicit --cta-runs/--mta-runs paths")
    if args.resamples < 2:
        parser.error("--resamples must be at least 2")
    if args.verify_predictions and not args.verify_baselines:
        parser.error("--verify-predictions also requires --verify-baselines")
    root = args.root.resolve()
    input_source = args.input_source or "custom"
    output = args.output or root / "analysis" / f"coverage_recalculation_{input_source}"
    audit = {"datasets": {}, "resamples": args.resamples, "seed": args.seed,
             "input_source": input_source,
             "bootstrap": "crossed_train_placement", "verified_from_processed_data": args.verify_baselines,
             "analysis_sha256": sha256(__file__)}
    runs, refs, contrasts, summaries, seed_means, missing = [], [], [], [], [], []
    for dataset in DATASETS:
        filename = ("coverage_matched_runs.csv" if dataset == "mta" and input_source == "matched"
                    else "coverage_runs.csv")
        path = (root / (getattr(args, f"{dataset}_runs") or Path(f"results_{dataset}/{filename}"))).resolve()
        d = validate_runs(pd.read_csv(path), dataset)
        source = path.relative_to(root) if path.is_relative_to(root) else path
        info = {"source": str(source), "sha256": sha256(path), "rows": len(d),
                "pairing_grid_complete": True, "duplicate_keys": 0,
                "reference_replicas_agree": True, "snaive7_coverage_invariant": True,
                "model_score_policy": "target_and_lag7_valid" if "model_score_policy" in d else "target_valid"}
        ref = d.drop_duplicates(CONDITION)[["dataset", *CONDITION, "snaive7", "common_mode"]].copy()
        status = "unverified"
        if args.verify_baselines:
            verified, gaps, hashes = verify_baselines(root, dataset, d)
            ref = ref.merge(verified, on=["dataset", *CONDITION], validate="one_to_one")
            missing.extend(gaps)
            status = "exact" if (verified.n_unmatched_cells == 0).all() else "mismatch"
            info["processed_sha256"] = hashes
            info["max_snaive7_abs_difference"] = float(verified.max_snaive7_abs_difference.max())
            info["max_common_mode_abs_difference"] = float(verified.max_common_mode_abs_difference.max())
            info["lag7_missing_target_entries_across_placements"] = len(gaps)
            info["unmatched_scoring_entries_across_placements"] = sum(g["included_in_model_score"] for g in gaps)
            if args.verify_predictions and "model_score_policy" in d:
                info.update(verify_predictions(root, d, ref))
        else:
            ref["score_support_status"] = status
        info["score_support_status"] = status
        audit["datasets"][dataset] = info
        d["MAE_over_snaive7"] = d.MAE / d.snaive7
        d["common_mode_over_snaive7"] = d.common_mode / d.snaive7
        d["score_support_status"] = status
        c = paired_contrasts(d).assign(dataset=dataset, score_support_status=status)
        s = summarise(c, args.resamples, args.seed).assign(dataset=dataset, score_support_status=status)
        for factor in KEY:
            means = c.groupby(["model", "metric", "contrast", factor], as_index=False).value.mean()
            means = means.rename(columns={factor: "seed"}).assign(factor=factor, dataset=dataset,
                                                                  score_support_status=status)
            seed_means.append(means)
        runs.append(d); refs.append(ref); contrasts.append(c); summaries.append(s)
    summary, references = pd.concat(summaries, ignore_index=True), pd.concat(refs, ignore_index=True)
    # All validation precedes writing: incomplete inputs cannot silently yield partial tables.
    output.mkdir(parents=True, exist_ok=True)
    tables = {"normalised_runs.csv": pd.concat(runs, ignore_index=True),
              "matched_snaive7.csv": references,
              "paired_contrasts.csv": pd.concat(contrasts, ignore_index=True),
              "summary.csv": summary, "seed_means.csv": pd.concat(seed_means, ignore_index=True),
              "lag7_missing_target_cells.csv": pd.DataFrame(missing, columns=["dataset", "placement_seed", "date", "node", "included_in_model_score"])}
    tables.update(publication_tables(tables["normalised_runs.csv"], tables["paired_contrasts.csv"],
                                     summary, input_source))
    for name, table in tables.items():
        table.to_csv(output / name, index=False, float_format="%.17g")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    audit_name = "support_audit.json" if args.verify_baselines else "csv_replay_audit.json"
    (output / audit_name).write_text(json.dumps(audit, indent=2) + "\n")
    (output / "REPORT.md").write_text(report(summary, references, audit, args.resamples, args.seed))
    (output / "training_seed_means.md").write_text(training_seed_report(tables["training_seed_means.csv"], input_source))
    headline = summary[(summary.metric == "MAE_over_snaive7") &
                       summary.contrast.isin(["shift_excess_penalty", "krin_reduction"])]
    print(headline[["dataset", "model", "contrast", "estimate", "ci_low", "ci_high", "positive",
                    "n_pairs", "score_support_status"]].to_string(index=False))
    print(f"\nWrote audited statistics to {output}")
    for dataset, info in audit["datasets"].items():
        if info["score_support_status"] != "exact":
            print(f"NOTE: {DATASETS[dataset]} scoring support: {info['score_support_status']}; "
                  "normalised values are not confirmed same-cell comparisons.")


if __name__ == "__main__":
    main()
