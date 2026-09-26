"""Check a fresh Appendix F.2 replay against values printed in the submitted PDF.

The independent reference records displayed values as strings, preserving each
table's precision. No training dependencies, raw data, or PDF reader are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def verify(output, reference, root):
    audit = json.loads((output / "audit.json").read_text())
    for key, expected in reference["bootstrap"].items():
        if audit.get(key) != expected:
            raise ValueError(f"Wrong analysis setting {key}: {audit.get(key)!r}")
    for path, expected in reference["inputs"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Manuscript input hash differs: {path}")
        entry = next((v for v in audit["datasets"].values() if v["source"] == path), None)
        if entry is None or entry["sha256"] != expected:
            raise ValueError(f"Replay did not use the manuscript input: {path}")
    code_hash = hashlib.sha256((root / "analysis/recalculate_coverage.py").read_bytes()).hexdigest()
    if audit["analysis_sha256"] != code_hash:
        raise ValueError("Replay audit refers to a different analysis version; run Appendix F.2 again")

    table4 = pd.read_csv(output / "table4.csv").set_index(["dataset", "backbone"])
    summary = pd.read_csv(output / "summary.csv")
    summary = summary[(summary.metric == "MAE_over_snaive7") &
                      (summary.contrast == "shift_excess_penalty")].copy()
    summary["dataset"] = summary.dataset.map({"cta": "Chicago", "mta": "Subway"})
    table48 = summary.set_index(["dataset", "model"])
    table49 = pd.read_csv(output / "training_seed_means.csv").set_index(
        ["dataset", "backbone", "train_seed"])
    table50 = pd.read_csv(output / "training_seed_mae.csv").set_index(
        ["dataset", "model", "train_seed", "shift", "coverage"])
    for name, frame, size in [("Table 4", table4, 4), ("Table 48", table48, 8),
                              ("Table 49", table49, 12), ("Table 50", table50, 96)]:
        if len(frame) != size or not frame.index.is_unique:
            raise ValueError(f"{name}: incomplete or duplicate output keys")
    if not table49.n_placements.eq(5).all() or not table50.n_placements.eq(5).all():
        raise ValueError("Every training-seed mean must average five placements")
    if not table4.input_source.eq("matched").all():
        raise ValueError("Table 4 must use the final matched inputs")

    counts = {tab: 0 for tab in reference["tables"]}

    def compare(tab, key, field, actual, expected):
        if isinstance(expected, str):
            decimals = len(expected.partition(".")[2])
            equal = f"{float(actual):.{decimals}f}" == expected
        else:
            equal = actual == expected
        if not equal:
            raise ValueError(f"Table {tab}, {key}, {field}: {actual!r} differs from PDF {expected!r}")
        counts[tab] += 1

    for row in reference["tables"]["4"]:
        key = (row["dataset"], row["backbone"])
        for field in ("D_uncentred", "D_KRIN", "R", "R_ci_low", "R_ci_high", "R_positive"):
            compare("4", key, field, table4.loc[key, field], row[field])
    for row in reference["tables"]["48"]:
        key = (row["dataset"], row["model"])
        for field in ("estimate", "ci_low", "ci_high", "positive"):
            compare("48", key, field, table48.loc[key, field], row[field])
    for row in reference["tables"]["49"]:
        key = (row["dataset"], row["backbone"], row["train_seed"])
        for field in ("D_uncentred", "D_KRIN", "R"):
            compare("49", key, field, table49.loc[key, field], row[field])
    for row in reference["tables"]["50"]:
        for shift in ("no shift", "regional shift"):
            for coverage in ("inside", "outside"):
                key = (row["dataset"], row["model"], row["train_seed"], shift, coverage)
                field = shift.replace(" ", "_") + "_" + coverage
                compare("50", key, "MAE", table50.loc[key, "MAE"], row[field])
    return {"status": "passed", "manuscript_sha256": reference["manuscript"]["sha256"],
            "comparison": "All displayed values at their printed precision; exact sign counts",
            "checked_values_by_table": counts, "checked_values_total": sum(counts.values()),
            "input_hashes_match": True, "analysis_sha256": code_hash,
            "raw_scoring_support_reverified": audit["verified_from_processed_data"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, default=root / "results/coverage/matched")
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).with_name("paper_tables_reference.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.output, json.loads(args.reference.read_text()), root)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
