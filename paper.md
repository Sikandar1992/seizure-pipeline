---
title: 'seizure-pipeline: a causal, leakage-resistant cross-validation toolkit for sparse-event forecasting, applied to epileptic seizure prediction'
tags:
  - Python
  - EEG
  - epilepsy
  - seizure prediction
  - machine learning
  - cross-validation
  - causal inference
  - time series
authors:
  - name: Sikandar Akbar
    affiliation: 1
  - name: Zahir Shah
    affiliation: 2
affiliations:
  - name: Department of Physics, University of Kashmir, Srinagar, India
    index: 1
  - name: Manipal Centre for Natural Sciences, Centre of Excellence, Manipal Academy of Higher Education, Manipal 576104, India
    index: 2
date: 25 September 2026
bibliography: paper.bib
---

# Summary

`seizure-pipeline` is a Python toolkit for building and evaluating machine-learning forecasters of rare, discrete events that emerge from an otherwise quiescent baseline, applied here to predicting epileptic seizures from intracranial EEG. It provides channel-aggregated and bivariate connectivity feature extraction for EEG clips, a block-level causal cross-validation scheme that prevents temporally correlated clips from a single seizure event from leaking between training and evaluation, a block-shuffle permutation test for statistical significance, and a five-seed stability check that flags results whose significance or effect size depends materially on an arbitrary random seed rather than on the data. The package is built around the American Epilepsy Society (AES) Kaggle Seizure Prediction Challenge dataset format, but its validation logic, the block-level causal splitting, the permutation test, and the seed-stability check, is domain-general and does not depend on any EEG-specific assumption; only the feature extraction and the definition of a non-splittable block are specific to this dataset. We have separately applied the same validation methodology to forecasting gamma-ray flares in blazars from Fermi-LAT light curves [@shahakbar2026], a physically unrelated problem with the same quiescent-baseline, sparse-event structure.

# Statement of need

Forecasting a rare, bursty event from a long record of ordinary behavior is a common task across scientific domains, and it is unusually easy to evaluate incorrectly. When nearby samples are highly correlated with each other, a train/test split that is not explicitly aware of that correlation can leak information from the evaluation period into training, producing a result that looks predictive but does not generalize. In EEG-based seizure prediction specifically, published evaluation protocols have been documented to be sensitive to exactly this failure mode, and small per-subject seizure counts (as few as four independent seizure events for some subjects in the AES Kaggle dataset) mean that a single arbitrary train/test split can also produce a result that is significant only by chance, in a way that does not survive re-splitting with a different random seed.

`seizure-pipeline` addresses both problems directly rather than as an afterthought. Its causal block-splitting logic groups the six sequential ten-minute clips belonging to one seizure's preictal hour into a single indivisible block, verifies this grouping against each clip's own metadata (the `sequence` field in the AES `.mat` files) rather than trusting a positional assumption silently, and never permits a block to appear on both sides of a train/test split. Its seed-sensitivity check repeats the entire cross-validation procedure under multiple random seeds and classifies each result as stable-significant, stable-null, or unstable, a check most seizure-prediction evaluation code does not perform and which, in our own use of this package, reclassified several results that a single-seed evaluation would have reported as positive findings without qualification. Researchers working on seizure prediction, or on any other sparse-event forecasting problem with a similar quiescent-baseline structure, can use this package directly on AES-format data or adapt its validation logic, which is deliberately separated from the EEG-specific feature extraction, to a different domain.

# Design and implementation

The package is organized around three stages. Feature extraction (`clip_features`) reduces each EEG clip to a fixed-length vector: per-channel spectral, temporal, and complexity features aggregated (mean, standard deviation, maximum) across channels, plus bivariate connectivity features (zero-lag and small-lag cross-correlation, phase-locking value in the alpha band) computed over a bounded, randomly subsampled set of channel pairs so that feature-extraction cost stays predictable regardless of a subject's channel count. Validation (`block_train_test_split`, `leave_one_preictal_block_out_cv`, `block_shuffle_permutation_test`) implements the causal, block-level splitting and significance testing described above. Reporting (`compute_fp_per_hour`, `sensitivity_at_target_fph`, `summarize_seed_stability`) converts raw predictions into the clinically relevant metrics used in the seizure-prediction literature (false alarms per hour, sensitivity at a fixed false-alarm target) rather than reporting only AUC, since a well-ranked but operationally unusable forecaster and a genuinely useful one can have similar AUC.

Two implementation details were driven directly by problems found while running the pipeline on real data rather than anticipated in advance, and are covered by the accompanying test suite: autocorrelation-based features are computed via FFT rather than a direct per-lag loop, since the direct computation's cost scales with sampling rate and became impractical at the 5000 Hz sampling rate of the dataset's human subjects though it was fast enough at the canine subjects' 400 Hz; and the connectivity feature computation explicitly frees each large intermediate array as soon as it is no longer needed, since holding all of them simultaneously exceeded available memory at the largest channel-count, highest-sampling-rate subject, a failure that a naive `del` of the array name alone did not fix because a NumPy view created during the same computation kept the underlying buffer alive regardless.

# Acknowledgements

We thank the American Epilepsy Society and the Kaggle Seizure Prediction Challenge organizers for making the underlying dataset available [@brinkmann2014].

# References
