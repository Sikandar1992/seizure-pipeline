# seizure-pipeline

A causal, leakage-resistant cross-validation toolkit for sparse-event
forecasting, applied here to predicting epileptic seizures from
intracranial EEG. It implements block-level causal train/test
separation, a block-shuffle permutation test, and a five-seed
stability check, for problems where a rare, discrete event must be
forecast from a long record of otherwise quiescent baseline data. The
same validation methodology has also been applied separately to
forecasting gamma-ray flares in blazars from Fermi-LAT light curves
([Shah & Akbar 2026](https://arxiv.org/abs/2605.07680)), a physically
unrelated problem with the same quiescent-baseline, sparse-event
structure.

## What this does

- **Feature extraction** (`clip_features`): reduces each 10-minute EEG
  clip to a channel-aggregated feature vector (band power, spectral
  edge, line length, kurtosis, autocorrelation decay) plus bivariate
  connectivity features (cross-correlation, phase-locking value) across
  a bounded, cost-capped set of channel pairs.
- **Causal block-level validation** (`block_train_test_split`,
  `leave_one_preictal_block_out_cv`): groups the six sequential clips
  belonging to one seizure's preictal hour into a single indivisible
  block, verified against each clip's own `sequence` metadata rather
  than trusted positionally, and never splits a block across train and
  test.
- **Statistical significance** (`block_shuffle_permutation_test`): a
  block-level permutation test, not a single point estimate.
- **Seed-sensitivity stability check** (`run_subject_multiseed`,
  `summarize_seed_stability`): repeats the full cross-validation under
  multiple random seeds and flags results whose significance or effect
  size depends materially on which seed was used, rather than on the
  data.
- **Clinical reporting** (`compute_fp_per_hour`,
  `sensitivity_at_target_fph`): false alarms per hour and sensitivity
  at a fixed false-alarm target, since AUC alone does not distinguish a
  well-ranked forecaster from an operationally usable one.

## Data

Built around the [AES Kaggle Seizure Prediction Challenge](https://www.kaggle.com/competitions/seizure-prediction)
`.mat` file format (electrode-by-time data matrix, sampling frequency,
channel names, and, for preictal clips, a `sequence` field). The
dataset itself is not included in this repository; download it from
Kaggle subject to the competition's data use terms, and point the
functions below at your local copy.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

```python
from pathlib import Path
from seizure_pipeline import run_all_subjects_multiseed, summarize_seed_stability

subject_dirs = {
    "Dog_1": Path("/path/to/Dog_1"),
    "Dog_2": Path("/path/to/Dog_2"),
    # ...
}

# Run the full causal cross-validation, 5 seeds, all 3 models, for every subject
results = run_all_subjects_multiseed(subject_dirs, seeds=(0, 1, 2, 3, 4))

# Collapse to one row per (subject, model): mean/std/range across seeds,
# how many seeds were individually significant, and a stability verdict
# (STABLE_SIGNIFICANT / STABLE_NULL / UNSTABLE)
summary = summarize_seed_stability(results)
print(summary)
```

## Testing

```bash
pytest test_seizure_pipeline.py -v          # fast tests only (~2s)
pytest test_seizure_pipeline.py --runslow   # includes Patient-scale checks (~60s)
```

The test suite covers the causal block-splitting guarantee, the
connectivity feature computation, a regression test for a memory-safety
fix at high channel count/sampling rate, and a correctness check for the
FFT-based autocorrelation implementation against a direct computation.

## Citing this software

If you use this package, please cite the JOSS paper (see `paper.md` /
`CITATION.cff`). The same validation methodology has also been applied
separately to blazar gamma-ray flare forecasting:

> Shah, Z. and Akbar, S. (2026). Advance warning of gamma-ray blazar
> flares from Fermi-LAT light curves: a strictly causal machine-learning
> backtest. arXiv:2605.07680.

## License

MIT. See `LICENSE`.
