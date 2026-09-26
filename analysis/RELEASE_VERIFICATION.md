# Submission verification

The manuscript reference is the PDF supplied on 2026-09-26, SHA-256
`7f3a0c69d4f731bc606975d2f895fc6806a8cc3a313702251e6f0f7aa1ef4421`.
The numerical reference in `paper_tables_reference.json` was extracted from
that PDF's printed tables, independently of the computation being checked.

## What is checked

| Manuscript table | Comparison | Values checked |
|---|---|---:|
| 4 | D before/after KRIN, R, R interval endpoints and positive-pair counts | 24 |
| 48 | Eight D estimates, interval endpoints and positive-pair counts | 32 |
| 49 | Twelve training-seed rows, each containing three five-placement means | 36 |
| 50 | 24 model/seed rows with MAE in four paired conditions | 96 |

All 188 comparisons pass at the precision printed in the manuscript. The
input SHA-256 values and the 20,000-draw, RNG-seed-0 crossed bootstrap settings
are also checked. The machine-readable result is
[`paper_verification.json`](../results/coverage/matched/paper_verification.json).

## Fresh-download procedure

1. Download **Full repo ZIP** from the
   [anonymous repository](https://anonymous.4open.science/r/KRIN-91E4/).
2. Extract it into a new directory. Do not add files or data from a development
   workspace. Install `requirements-statistics.txt` in a fresh Python 3.10+
   environment.
3. From the extracted root, run the exact Appendix F.2 commands:

   ```bash
   python3 analysis/analyse_coverage.py --input-source matched \
     --output results/coverage/matched
   python3 -m unittest discover -s analysis -p test_coverage_statistics.py -v
   ```

4. Check the outputs against all four manuscript tables:

   ```bash
   python3 analysis/verify_paper_tables.py --output results/coverage/matched
   ```

The four portable tests check missing/duplicate inputs, normalisation, shared
placement resampling, and five-placement averaging. The table verifier fails
if any printed effect, interval endpoint, sign count or MAE differs. It also
rejects the original-input sensitivity analysis as the final table input.
An exit code alone is insufficient: the expected files and values must exist.

CSV-only replay writes an unverified raw-support status and a fresh
`csv_replay_audit.json`; it leaves the historical `support_audit.json` intact.
The latter records the earlier processed-data and prediction checks. This
release check does not retrain models or rerun those data-level checks.

## Synchronising the anonymous copy

The anonymous ZIP inspected on 2026-09-26 still represented its 2026-09-24
snapshot. It contained the old `analyse_coverage.py` and neither final input
CSV. The Appendix F.2 command printed "no run file" for both datasets while
returning zero; no final statistics were generated.

Updating the source Git repository and updating the anonymous snapshot are
separate steps. After publishing this release, the anonymous repository owner
must refresh its snapshot if it does not update automatically. Download the
anonymous ZIP again, check that its files match `TABLE4_UPDATE_MANIFEST.json`,
and repeat the commands above. Availability on the source Git host alone does
not establish availability at the reviewer-facing URL.
