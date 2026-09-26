# Coverage analysis for Table 4

`analyse_coverage.py` calls `recalculate_coverage.py`. The calculation uses a
complete grid of four model configurations, training seeds 0–2, placement seeds
0–4, two shift conditions, and Inside/Outside observation coverage. Missing or
duplicate keys, inconsistent baseline replicas, nonfinite errors, and zero
SNaive-7 denominators are rejected.

The final manuscript uses `--input-source matched`; the `original` selection is
retained only for sensitivity and provenance.

This repository root contains the accompanying package described in Appendix
F.2; the analysis code and input CSVs are not in a separate supplement.

To compare a fresh CSV-only replay with the submitted manuscript, run
`python3 analysis/verify_paper_tables.py --output results/coverage/matched`.
The reference file `paper_tables_reference.json` records the PDF's SHA-256 and
the printed values in Table 4 and Tables 48–50. Verification uses each printed
value's decimal precision, checks all 96 condition-specific MAE means, and
requires the final matched input hashes and bootstrap settings.

The separate [matched-rerun entry point and configuration](MATCHED_RERUN.md)
document how the final Subway errors were generated. CSV-only replay does not
rerun that training or recheck raw scoring support.

## Input files

| Selection | Chicago | Subway |
|---|---|---|
| `--input-source original` | `results_cta/coverage_runs.csv` | `results_mta/coverage_runs.csv` |
| `--input-source matched` | `results_cta/coverage_runs.csv` | `results_mta/coverage_matched_runs.csv` |

Each file has 240 evaluation rows. The `snaive7` column is the baseline error
for the same placement, shift, and observation condition. It does not acquire
independent repetitions from model or training-seed labels. The analyser checks
that repeated reference errors agree and that SNaive-7 is invariant to
Inside/Outside coverage.

The matched Subway file is a new evaluation of 12 fitted models on cells with
both a valid target and a valid lag-7 reference. The original file scores 38
additional target-valid entries across five placements for learned models only.
Its normalised statistics can be computed, but they do not compare identical
scoring supports. The two files' SPIN-s results also differ because the matched
evaluation required fresh training. `MAE_all_valid`, `previous_MAE`, and
`replay_MAE_difference` retain that comparison. Select the input set used by the
manuscript version being checked; the script never silently substitutes a rerun.

## D, R, and the resampling unit

Let `a` be a training seed, `b` a placement seed, `z` the shift condition, and
`c` the Inside/Outside observation condition. For each model `m`:

```text
q[m,a,b,z,c] = MAE[m,a,b,z,c] / SNaive7[b,z,c]
p[m,a,b,z]   = q[m,a,b,z,outside] - q[m,a,b,z,inside]
D[m,a,b]     = p[m,a,b,shift] - p[m,a,b,no_shift]
R[a,b]       = D[uncentred,a,b] - D[KRIN,a,b]
```

These quantities are dimensionless. Division and pairing occur within each run
before taking any mean. A mean of per-run ratios is not replaced by a ratio of
mean MAEs. Raw-MAE contrasts are retained separately for inspection.

For each of 20,000 bootstrap replicates, sample three training-seed labels with
replacement and independently sample five placement labels with replacement.
Evaluate the Cartesian product of these draws and average its paired contrast.
The same placement draw is shared across sampled training seeds, and the same
draws are reused across model/condition comparisons. Report the 2.5th and 97.5th
percentiles with RNG seed 0. This is a crossed-factor bootstrap, not resampling
15 flattened pairs and not independently resampling placements within each
training seed.

There are only three fitted training seeds and five sampled placements, so the
intervals are exploratory. Sign counts describe crossed evaluation pairs and
are not counts of independent training runs.

## Files to inspect

Run either README command to produce:

- `table4.csv`: four dataset/backbone rows, with `D_uncentred`, `D_KRIN`, `R`,
  crossed-factor 95% intervals, and positive-pair counts.
- `training_seed_means.csv` and `.md`: 12 rows. Each dataset/backbone/training-seed
  row averages the five placement-specific D/R values. The CSV includes
  `raw_D_uncentred`, `raw_D_KRIN`, and `raw_R` in original MAE units as well.
- `training_seed_mae.csv`: 96 rows. For each dataset/model/training seed/shift/
  coverage cell, the mean MAE and mean per-run MAE/SNaive-7 ratio across five
  placements are reported separately.
- `paired_contrasts.csv`: all unaggregated paired contrasts.
- `matched_snaive7.csv`: one baseline row per dataset/placement/shift/coverage.
- `audit.json`: the selected input paths, SHA-256 hashes, seed settings, and checks.
- `support_audit.json`: preserved data-level audit from the original evaluation;
  a CSV-only replay does not overwrite this record.
- `csv_replay_audit.json`: the independent CSV-only statistical replay, which
  explicitly does not reverify raw-data scoring support.

CSV-only runs label the scoring-support audit as unverified: they do not reread
the raw observations. When the processed data are available, `--verify-baselines`
recomputes training-free errors and checks their scoring support. The optional
`--verify-predictions` additionally requires the saved matched-rerun predictions;
those large files are not part of this scoped code release.

The portable statistical checks need only NumPy and pandas. From the repository
root, run:

```bash
python3 -m unittest discover -s analysis -p test_coverage_statistics.py -v
```

They check that missing or duplicate pairs fail, pure scale contraction vanishes
after per-run normalisation, placement draws are shared across training seeds,
and each training-seed export averages five per-run values.
