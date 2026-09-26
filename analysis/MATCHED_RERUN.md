# Final Subway matched-support rerun

The final Table 4 and Tables 48–50 use the retained Chicago original records and
`results_mta/coverage_matched_runs.csv`. The Subway file contains a new fit for
each of `spin`, `spin_c`, `ignnk_off`, and `ignnk_off_c` at training seeds 0–2.
It is not an arithmetic adjustment to the original fitted SPIN-s errors.

## Entry point and recorded sources

- Runnable release entry point: [experiments/run_coverage_matched.py](../experiments/run_coverage_matched.py).
- Configuration snapshot: [configs/coverage_matched/mta.yaml](../configs/coverage_matched/mta.yaml).
  Its contents are identical to `configs/mta.yaml` used for the final run.
- Exact executed runner, retained for its provenance hash:
  [experiments/provenance/run_coverage_matched_original.py](../experiments/provenance/run_coverage_matched_original.py).
  Its SHA-256 is `a14d5848eaf595318644ff8d1c0bfd813b8f467b866f4698c57e5431687f4603`,
  as recorded in [the original provenance](../results_mta/coverage_matched_provenance.json).
  This archive uses the original workspace module names; use the runnable
  release entry point above.
- [Recorded environment](../results_mta/coverage_matched_environment.json) and
  [source mapping](../results_mta/coverage_matched_source_map.json).

The release entry point maps the original `run_infoexp` import to the released
`run_history` module. The training and scoring functions are unchanged. Its
additional output-directory option keeps new results separate from the retained
CSV; its provenance now also records the configuration and experiment-module
hashes. The source mapping distinguishes original hashes from release hashes.
Shared training and model source edits are documentation changes and removal of
unused model aliases, with no change to the four model configurations above.

## Training and evaluation settings

Training uses the unshifted `inregime` scenario, geographic graph `A_geo`, random
target masks at ratio 0.5, and training seeds 0, 1, 2. The configuration records
60 epochs, batch size 32, learning rate 0.001, weight decay 0.00001, patience 12,
hidden dimension 64, two layers and dropout 0.1 for the simplified model. The
released IGNNK adapter applies its own `HP` overrides in
`src/models/official_wrap.py`: learning rate 0.0001 and batch size 4.
`train_one` uses those overrides for both IGNNK-R configurations.

Each fitted model is evaluated across placement seeds 0–4, with paired
Inside/Outside observation sets in both no-shift and local-shift conditions.
The local multiplier is 0.25 over a 30% graph region, including the preceding
14-day history. Model MAE and SNaive-7 both require a valid target and valid
lag-7 value. Training loss and input masks retain the original design.
`MAE_all_valid` separately records the original target-valid-only scoring rule.
The evaluation batch-size limit is 16, and the runner defaults to four CPU
threads and a CUDA memory fraction of 0.1.

## Run from the released layout

This is a **training** command, separate from the NumPy/pandas-only command in
Appendix F.2. It requires the full training dependencies from `requirements.txt`
and the processed Subway observations, graphs and splits produced by the data
pipeline. Those data and model checkpoints are not bundled. The original
`results_mta/coverage_runs.csv` is bundled and supplies the comparison errors.

From the repository root, an equivalent command for a fresh run is:

```bash
python3 experiments/run_coverage_matched.py configs/coverage_matched/mta.yaml \
  --models spin,spin_c,ignnk_off,ignnk_off_c --require-cuda \
  --threads 4 --cuda-memory-fraction 0.1 --eval-batch-size 16 \
  --output-dir results_mta/reproduced_matched
```

The selected GPU is controlled by `CUDA_VISIBLE_DEVICES`. Use a device available
in the new environment. Add `--resume` only to continue this new output
directory's own checkpoints and complete rows. The runner refuses to overwrite
an existing result without that flag.

Outputs in the selected directory are `coverage_matched_runs.csv`,
`coverage_matched_provenance.json`, `coverage_matched_checkpoints/`, and
`coverage_matched_predictions/`. To analyse newly generated errors separately:

```bash
python3 analysis/analyse_coverage.py \
  --cta-runs results_cta/coverage_runs.csv \
  --mta-runs results_mta/reproduced_matched/coverage_matched_runs.csv \
  --output results/coverage/reproduced_matched
```

The final recorded environment used Python 3.10.18, PyTorch 2.7.1+cu118,
NumPy 2.1.2, pandas 2.3.2 and CUDA 11.8. Deterministic algorithms were not
enabled, and a fresh fit can differ across environments. The fixed released
CSVs reproduce the manuscript statistics; the code and settings document the
training procedure without claiming bit-identical fresh training. No new
training run is required to verify Appendix F.2.
