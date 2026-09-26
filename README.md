# Level Shift, Not Aggregation

This repository provides the data-processing, training, and analysis code used to reproduce the main-paper tables, figures, and central quantitative claims.

Two claims organise the code.

1. When evaluation demand shifts beyond the training range, learned
   reconstruction models can have higher error than training-free seasonal
   persistence and common-mode correction.
2. Subtracting each node's recent history mean and restoring it after prediction
   reduces error across simplified and released implementations without adding
   learned parameters. Its advantage over a strong training-free baseline is
   conditional; the controlled coverage and transition experiments identify
   both benefits and limitations.

## Layout

```
configs/          one YAML per dataset; paths, splits, masking, optimisation
src/              library code
  utils.py            config, seeding, scaling, masking, metrics, geometry
  data_prep*.py       raw feeds to a (days x nodes) matrix, one file per dataset
  build_graphs.py     geographic and correlation adjacency
  splits.py           the Normal and Shock scenarios, scalers fit on train rows
  kriging_data.py     the task: history window, target-day mask, interventions
  models/             backbones, the KRIN wrapper, released-code adapters
  train.py            shared training loop
  evaluate.py         per-regime evaluation on a fixed mask
  baselines.py        training-free predictors
  shift_amp.py        the level-shift diagnostic
experiments/      runnable experiments, one file each
analysis/         summarisers that turn run files into the reported numbers
```

## What is here

Code and configurations cover the main-paper experiments. The small per-run
coverage CSVs for Chicago and Subway are included so the coverage statistics
and training-seed means can be recomputed without model training or raw data.
Raw datasets, model checkpoints, and the complete set of Appendix experiment
outputs are not included. Additional outputs are written to the directories
named under `paths` in each configuration.

## Setup

```
pip install -r requirements.txt
```

`torch_scatter` has no universal wheel; if the plain install fails, install the
build matching your PyTorch/CUDA from https://data.pyg.org/whl/. It, `einops` and
`torch_geometric` are needed only for the released-implementation adapters
(model names ending in `_off`). The Table 4 statistics below need only NumPy and pandas.

A CUDA GPU is assumed but not required; `train.device` in each config selects the
device. The runs reported in the paper used a single 24 GB card, and the largest
single training run takes about half an hour.

## Data

Every dataset is public and downloadable without credentials. Place the raw
files where the config expects them and build the derived artefacts:

```
python3 src/data_prep_bike.py            # Capital Bikeshare
python3 src/data_prep_cta.py             # Chicago Transit Authority
python3 src/data_prep.py                 # Airports (EUROCONTROL)
python3 src/data_prep_mta.py             # NYC Subway
python3 src/data_prep_aqs.py             # EPA NO2, the held-out domain

python3 src/build_graphs.py  --config configs/<name>.yaml
python3 src/splits.py        --config configs/<name>.yaml
```

This writes, under the config's `paths.processed`, a traffic matrix, the
adjacency matrices, the split index and the fixed evaluation masks. Downstream
experiments share these files and use explicit training seeds. Exact numerical
replays can still depend on the software and hardware environment.

## Task

A day `t` is evaluated by hiding a set of nodes' values on that day only. Each
model receives three inputs and nothing else:

```
H_t = [x_{t-K}, ..., x_{t-1}]      the K-day history, unmasked for every node
x_t * m                            the target day with hidden nodes zeroed
m                                  the mask channel
```

The history of a hidden node stays visible, which is what makes a training-free
persistence prediction available to every method. Loss and metrics are computed
on hidden nodes only.

KRIN subtracts each node's own K-day window mean from both history and observed
target day, runs the backbone, and adds the level back. It introduces no learned
parameters: `ignnk` and `ignnk_c` have identical parameter counts.

## Running the experiments

All commands are run from this directory. Results are written under each config's
`paths.results` as one row per run.

```
# main arm: six backbones with and without KRIN, both conditions, three ratios
python3 experiments/run_main.py configs/cta.yaml --tag=geo3m10 \
        stgnn stgnn_c ignnk ignnk_c satcn satcn_c \
        grin grin_c spin spin_c kits kits_c

# released implementations at the hyperparameters their own papers report
python3 experiments/run_main.py configs/cta.yaml --tag=official2 --ratios=0.5 \
        ignnk_off ignnk_off_c satcn_off satcn_off_c grin_off grin_off_c \
        kits_off kits_off_c spin_off spin_off_c

# controlled level-shift sweep, applied at evaluation only (Appendix C.2)
# main sweep: 12 alphas {0.15,...,5.0}, shift also covers the K history days
python3 experiments/run_shift_sweep.py configs/cta.yaml --mode=pre     # shiftsweep.csv
# onset variant: alpha {0.25, 0.5, 2.0}, shift starts on the first evaluation day
python3 experiments/run_shift_sweep.py configs/cta.yaml --mode=onset   # shiftsweep_onset.csv

# incomplete histories (Appendix C.1): 25% point holes, or 7-day mean blocks.
# Add --split-shock-years for Airports.
python3 experiments/run_main.py configs/cta.yaml --tag=protoB \
        --scenarios=cross_pre2shock --ratios=0.5 \
        --hist-missing=0.25 --hist-pattern=point \
        stgnn stgnn_c ignnk ignnk_c satcn satcn_c grin grin_c spin spin_c kits kits_c
python3 experiments/run_main.py configs/cta.yaml --tag=protoB_block \
        --scenarios=cross_pre2shock --ratios=0.5 \
        --hist-missing=0.25 --hist-pattern=block --hist-block-len=7 \
        stgnn stgnn_c ignnk ignnk_c satcn satcn_c grin grin_c spin spin_c kits kits_c

# training-free baselines, written to baselines.csv
python3 src/baselines.py --config configs/cta.yaml

# observation coverage, with the no-shift control
python3 experiments/run_coverage.py configs/cta.yaml spin,spin_c
python3 experiments/run_coverage.py configs/cta.yaml ignnk_off,ignnk_off_c

# The coverage runner writes the original scoring design. To regenerate the
# final Table 4 statistics, use the retained matched CSVs below.

# regional vs global shift, and history freshness
python3 experiments/run_history.py configs/cta.yaml

# the first days after a transition, one day at a time
python3 experiments/run_transition.py configs/cta.yaml

# shift detection against baseline-relative reversal, day by day
python3 experiments/run_alarm.py configs/cta.yaml

# level-equivariance probe
python3 experiments/run_probe.py configs/cta.yaml

# backbone chosen on validation rather than on the evaluation split
python3 experiments/run_valselect.py configs/cta.yaml

# training-free baselines on METR-LA under the point-missing protocol
python3 experiments/run_metrla.py data/external/metr-la.h5
```

Each experiment trains once per (model, training seed) and applies its
intervention at evaluation, so a sweep costs one training run per arm.

## Turning runs into reported numbers

The coverage analysis can read the included CSVs directly. Other analysis
commands require the corresponding experiment outputs from the steps above.

```
python3 analysis/analyse_coverage.py --input-source matched --output results/coverage/matched
python3 analysis/bootstrap_ci.py         # mask-cluster and hierarchical intervals
python3 analysis/monitor_alarm.py        # rolling-window monitor alarm rates
python3 analysis/neighbour_mean.py       # neighbour mean in three spaces
python3 analysis/probe_history.py        # what the model receives as history
python3 analysis/probe_fallback.py       # how often a window has no observation
python3 analysis/fidelity_table.py       # released-implementation results table
python3 analysis/make_figures.py         # every figure, plus its numbers as CSV
```

`analysis/make_figures.py` writes each figure alongside a CSV of exactly the
values it plotted, so any number in a figure can be traced to the run file it
came from.

## Seeds and replication

Training seeds are 0, 1, 2 and mask seeds 0 to 9, giving 30 evaluations per
reported cell in the main arm. Training seeds default to 0–2 in the runner; mask seeds come from
the dataset configurations. The commands above use these defaults; `splits.py` must be rerun if
`mask.seeds` is changed, since it is what writes the evaluation masks.

Training and mask randomness are separate knobs: `train_seed` drives weight
initialisation, batch order and the dynamic training masks, while the mask seed
selects a fixed evaluation mask from the split files. Evaluations that share a
training seed reuse one fitted model and are not independent training runs,
which is why intervals are clustered rather than taken over all 30.

The coverage experiment adds a placement seed, 0 to 4, that redraws the target
set and both observed sets. This matters: its observed sets are constructed
inside the experiment rather than read from the mask files, so the mask seed does
not vary anything there and the placement seed is what provides replication.
`analyse_coverage.py` forms per-run ratios to the same-condition SNaive-7 error
before forming paired contrasts. It independently resamples training seeds and
placement labels, then takes their Cartesian product, retaining the same draws
across all paired models and conditions. The 15 combinations are not independent
replications. Intervals are exploratory with only 3 training seeds and 5
placements; sign counts are descriptive.

## Table 4 and training-seed means

The final manuscript Table 4 uses **matched** inputs: original Chicago runs and
the separately retained Subway matched-scoring rerun. The original Subway arm is
a sensitivity comparison and must not replace the final Table 4 input.

The coverage analysis requires an explicit input selection. Both commands below
run on the included CSVs using NumPy and pandas, without GPU training:

```bash
# Both original coverage_runs.csv files
python3 analysis/analyse_coverage.py --input-source original --output results/coverage/original

# Chicago original and the separately retained Subway matched-scoring rerun
python3 analysis/analyse_coverage.py --input-source matched --output results/coverage/matched
```

| Input set | Normalised D/R and crossed-factor intervals | Training seeds 0–2, each averaged over five placements | Original-scale MAE by training seed and condition |
|---|---|---|---|
| Original | [Table CSV](results/coverage/original/table4.csv) | [Readable table](results/coverage/original/training_seed_means.md), [CSV](results/coverage/original/training_seed_means.csv) | [MAE CSV](results/coverage/original/training_seed_mae.csv) |
| Matched scoring | [Table CSV](results/coverage/matched/table4.csv) | [Readable table](results/coverage/matched/training_seed_means.md), [CSV](results/coverage/matched/training_seed_means.csv) | [MAE CSV](results/coverage/matched/training_seed_mae.csv) |

For each run, define `q = MAE / SNaive7` using its own same-condition baseline.
Then `D = (q_out - q_in)_shift - (q_out - q_in)_no_shift` and
`R = D_uncentred - D_KRIN`. The implementation forms those paired quantities
before averaging and takes 20,000 crossed-factor bootstrap draws (seed 0).
The CSVs also retain raw-MAE contrasts in separately named columns.

These input sets must not be interchanged when matching a manuscript table.
In the original Subway results, 38 of 95,184 target-valid entries across the
five placements have a missing lag-7 value and are excluded only by SNaive-7.
The matched-scoring file comes from 12 fresh training runs evaluated on the
common valid targets; its SPIN-s errors differ from the original fitted runs.
The original files remain available. [Table 4 protocol and inputs](analysis/TABLE4.md)
give the pairing, source hashes, output columns, and scoring caveat.
Recorded data-level checks are preserved in each result directory as
`support_audit.json`. A CSV-only replay reproduces the numerical statistics
while marking its own scoring-support check as unverified; this does not rerun
the recorded raw-data or saved-prediction checks. Those optional checks require
processed data and prediction files that are outside this scoped release.

## Interventions

`src/kriging_data.py` holds the switches the controlled experiments use. All
default to the untouched protocol.

```
set_level_shift(alpha, onset, nodes=None)   multiply the evaluation span by alpha;
                                            nodes restricts it to a subset, onset
                                            chooses whether the K days before the
                                            span move with it
set_history_staleness(days)                 blank the most recent `days` slots of
                                            the hidden nodes' own window and fill
                                            them with the mean of what remains
set_hist_missing(rate, pattern, seed)       point or block holes in the history,
                                            the same holes for every model
```

The hole mask is seeded from a digest of its parameters rather than from
`hash()`, so it is identical across processes and machines; every model in a
comparison sees the same holes, and a rerun reproduces them.

Two details matter when reading results that use them. The window is always
`t-K` to `t-1`; it is never moved into the past. And filling a gap with the mean
of the remaining observations leaves the window mean unchanged, so the level a
centred model computes equals the mean of the observations that actually
survive. When none survive, the fill is zero in scaled space, which is the node's
training-split mean, and centring has nothing left to adapt with.
