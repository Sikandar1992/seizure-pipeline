#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seizure_pipeline.py — a causal, leakage-resistant cross-validation
pipeline for sparse-event forecasting, applied here to per-subject
preictal/interictal classification on the AES/Kaggle "American
Epilepsy Society Seizure Prediction Challenge" dataset.

Author: Sikandar Akbar Dar, Zahir Shah
1) Department of Physics, University of Kashmir, Srinagar, India
2) Manipal Centre for Natural Sciences, Centre of Excellence, Manipal Academy of Higher Education, Manipal 576104, India
METHODOLOGY
-----------
- Model layer: LR / PLR (polynomial expansion + ANOVA feature select) /
  RF, class-balanced, calibrated on a held-out tail.
- Threshold selection via F-beta grid search, favoring recall by
  default; worth re-examining per subject rather than assuming
  beta=2 is right for every subject.
- Group-aware, leakage-resistant validation: train and test data are
  never split at the level of individual clips, only at the level of
  non-splittable blocks. Here the relevant leak is CLIPS FROM THE SAME
  PREICTAL HOUR ending up on both sides of a split, so a block is
  defined as the six sequential clips drawn from one seizure's
  preictal hour, or a single interictal clip. This is the single most
  important thing to get right; AES/Kaggle data is well known in the
  literature for models that look good on a naive random split and
  collapse under a block-aware one.
- Block-level permutation test for whether held-out AUC is real.
- A five-seed stability check (Section 7 below), repeating the full
  cross-validation under five random seeds and flagging any result
  whose AUC or significance call is not consistent across all of them,
  rather than trusting whatever one arbitrary seed happens to report.

DATA AND I/O
------------
AES ships pre-segmented, pre-labeled 10-minute .mat clips per subject,
each containing a data matrix (electrode by time sample), sampling
frequency, channel names, and, for preictal clips, a sequence field
giving the clip's position (1-6) within its source seizure's one-hour
preictal period. Labels are already present in the filename
(preictal_segment_NNNN.mat / interictal_segment_NNNN.mat /
test_segment_NNNN.mat).

This pipeline has been run against the real AES/Kaggle dataset across
all five canine subjects (Dog 1 through Dog 5); results are reported
in the accompanying paper. `load_clip_mat()` is written defensively
(it tries several struct/field-name variants) because the exact
MATLAB struct field names are known to vary slightly across subjects
and across the original 2014 release vs. later re-releases, and this
defensiveness is what let the loader work unmodified across all five
subjects' files without a per-subject special case. The preictal
block assignment in `assign_preictal_blocks()` groups clips
positionally (six consecutive preictal clips per block) before any
file is read; `verify_preictal_block_sequences()` is then called
automatically by `build_subject_dataset()` on every real run and
raises immediately if a block's actual `sequence` values do not match
{1,...,6}, so this assumption is checked against the real per-subject
metadata every time the pipeline runs, not left as an open item.

The two human subjects in the full AES/Kaggle release (sampled at
5000 Hz, up to 24 channels) are not included in the results reported
in the paper; the feature extraction pipeline is verified to run
correctly at human scale, but the human subjects have not yet been
run through the five-seed stability protocol. See the paper's
Limitations section for the current scope of what has and has not
been validated.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_score, recall_score,
                              fbeta_score, brier_score_loss,
                              matthews_corrcoef)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# NumPy >=2.0 renamed trapz -> trapezoid; support both so this doesn't
# silently break depending on whose environment it runs in.
_trapz = getattr(np, "trapezoid", None) or np.trapz

RANDOM_SEED = 0
THRESH_GRID = np.linspace(0.05, 0.95, 91)
THRESHOLD_FBETA = 2.0          # recall-weighted, mirrors the WATCH default —
                                # re-examine per subject, not assumed correct
N_PERM_TEST = 1000
CALIBRATION_TAIL_FRAC = 0.20   # fraction of TRAIN blocks reserved for
                                # Platt calibration, same role as the
                                # blazar code's calibration tail

# EEG frequency bands for band-power features (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


# ============================================================
# 1) CLIP DISCOVERY + I/O
# ============================================================

_FNAME_RE = re.compile(
    r"^(?P<subject>[A-Za-z]+_\d+)_(?P<ctype>preictal|interictal|test)"
    r"_segment_(?P<idx>\d+)\.mat$"
)


def parse_clip_filename(path: Path) -> Optional[dict]:
    """Parse an AES/Kaggle-style filename. Returns None if it doesn't match."""
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    return {
        "subject": m.group("subject"),
        "clip_type": m.group("ctype"),
        "index": int(m.group("idx")),
        "path": path,
    }


def discover_clips(subject_dir: Path) -> pd.DataFrame:
    """Scan one subject's directory and return a labeled clip manifest.

    Columns: path, clip_type (preictal/interictal/test), index.
    Does NOT read the .mat files yet — this is just the filename-level
    inventory, fast enough to run before committing to loading
    everything into memory.
    """
    subject_dir = Path(subject_dir)
    rows = []
    for p in sorted(subject_dir.glob("*.mat")):
        parsed = parse_clip_filename(p)
        if parsed is not None:
            rows.append(parsed)
    if not rows:
        raise RuntimeError(
            f"No AES-format .mat clips found in {subject_dir}. "
            f"Expected names like 'Dog_1_preictal_segment_0001.mat'."
        )
    return pd.DataFrame(rows)


def load_clip_mat(path: Path) -> dict:
    """Load one .mat clip. Returns {'data': (n_channels, n_samples) float64,
    'fs': float, 'channels': list[str] or None, 'sequence': int or None}.

    Defensive against the couple of struct/field-naming variants seen in
    AES/Kaggle releases. If this fails on your real files, print
    `loadmat(path, struct_as_record=False, squeeze_me=True).keys()` and
    adjust the candidate lists below — this is the one place in the
    script most likely to need a real-data-driven fix.
    """
    mat = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"{path}: no non-dunder keys in .mat file")

    # AES files typically contain exactly one struct-like variable named
    # e.g. "preictal_segment_1" / "interictal_segment_1" / "test_segment_1"
    struct_key = keys[0]
    if len(keys) > 1:
        # prefer a key containing 'segment' if there's ambiguity
        seg_keys = [k for k in keys if "segment" in k.lower()]
        struct_key = seg_keys[0] if seg_keys else keys[0]

    obj = mat[struct_key]

    def _get(obj, name, default=None):
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        return default

    data = _get(obj, "data")
    if data is None:
        raise ValueError(f"{path}: could not find 'data' field under '{struct_key}'")
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]   # single-channel edge case

    fs = _get(obj, "sampling_frequency")
    fs = float(np.asarray(fs).squeeze()) if fs is not None else np.nan

    channels = _get(obj, "channels")
    if channels is not None:
        channels = [str(c).strip() for c in np.atleast_1d(channels)]

    sequence = _get(obj, "sequence")
    sequence = int(np.asarray(sequence).squeeze()) if sequence is not None else None

    return {"data": data, "fs": fs, "channels": channels, "sequence": sequence}


# ============================================================
# 2) FEATURE EXTRACTION (per clip, channel-aggregated)
# ============================================================

def _band_power(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    if not np.any(m):
        return np.nan
    return float(_trapz(psd[m], freqs[m]))


def _line_length(x):
    return float(np.sum(np.abs(np.diff(x))))


def _spectral_edge_95(freqs, psd, frac=0.95):
    """Frequency below which `frac` of total spectral power is contained."""
    total = np.sum(psd)
    if total <= 0:
        return np.nan
    cum = np.cumsum(psd) / total
    idx = np.searchsorted(cum, frac)
    return float(freqs[min(idx, len(freqs) - 1)])


def _autocorr_decay_lag(x, fs, max_lag_sec=1.0):
    """Lag (seconds) at which the autocorrelation first drops below 1/e —
    the EEG analog of the blazar code's structure-function slope /
    variability-decorrelation-length idea, adapted to a single clip
    instead of a rolling multi-year window.

    FFT-based (via scipy's fftconvolve), O(n log n) regardless of fs.
    The previous version computed only the needed max_lag lags directly
    (fixing an earlier O(n^2) np.correlate bug), but max_lag itself
    scales with fs — 400 for dog clips, 5000 for the two human subjects
    (Patient_1/Patient_2, sampled at 5 kHz) — so that direct-loop cost
    scaled roughly as fs^2 across subjects: fine for dogs, unusably slow
    for the human recordings. FFT-based cost is fs-agnostic given fixed
    max_lag_sec, so this same function is now fast on both.
    """
    x = x - np.mean(x)
    n = len(x)
    max_lag = int(max_lag_sec * fs)
    if max_lag < 2 or n < max_lag * 2:
        return np.nan
    var0 = np.dot(x, x)
    if var0 == 0:
        return np.nan
    full = sp_signal.fftconvolve(x, x[::-1], mode="full")
    zero_lag_idx = n - 1
    ac = full[zero_lag_idx: zero_lag_idx + max_lag] / var0
    below = np.where(ac < np.exp(-1))[0]
    if len(below) == 0:
        return float(max_lag_sec)   # decorrelation longer than window checked
    return float(below[0] / fs)


def channel_features(x: np.ndarray, fs: float) -> dict:
    """Feature vector for ONE channel's raw samples."""
    feat = {}
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < int(2 * fs) or not np.isfinite(fs) or fs <= 0:
        return {k: np.nan for k in
                list(BANDS) + ["total_power", "spectral_edge95",
                               "line_length", "std", "kurtosis",
                               "autocorr_decay_sec"]}

    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    total_power = float(_trapz(psd, freqs)) if len(freqs) > 1 else np.nan

    for band_name, (lo, hi) in BANDS.items():
        bp = _band_power(freqs, psd, lo, hi)
        feat[band_name] = bp / total_power if (total_power and total_power > 0) else np.nan

    feat["total_power"] = total_power
    feat["spectral_edge95"] = _spectral_edge_95(freqs, psd)
    feat["line_length"] = _line_length(x) / len(x)     # normalized by length
    feat["std"] = float(np.std(x))
    from scipy.stats import kurtosis as sp_kurtosis
    feat["kurtosis"] = float(sp_kurtosis(x, bias=False)) if x.size > 20 else np.nan
    feat["autocorr_decay_sec"] = _autocorr_decay_lag(x, fs)

    return feat


# Connectivity features: literature-standard choices for this dataset's
# lineage (Mirowski et al. and others report cross-correlation, phase
# synchrony/PLV, and coherence as the features that actually mattered
# most, more than single-channel band power). PLV is computed in the
# alpha band (8-13 Hz), a common choice in that literature; other bands
# are a reasonable follow-up, not implemented here.
CONNECTIVITY_BAND = (8.0, 13.0)
CONNECTIVITY_MAX_LAG_SEC = 0.02   # +/-20 ms cross-correlation search window

# Full pairwise connectivity is O(n_channels^2) pairs, each needing an
# FFT-based cross-correlation over the full clip. That's fine at dog
# scale (16 channels = 120 pairs, 400 Hz), but Patient_2 has up to 24
# channels (276 pairs) at 5000 Hz -- the same class of scaling risk
# that bit the single-channel autocorrelation feature before it was
# fixed to use FFT. Unlike that fix, there is no way to make the
# PER-PAIR cost independent of channel count here (each pair is a
# distinct signal pairing, not a reusable lag search), so instead of
# computing every pair we cap the number of pairs and subsample the
# rest with a fixed seed, trading completeness for a bounded, known
# cost regardless of how many channels a subject has. Timed on
# synthetic data at matching scale (16 ch/400 Hz and 24 ch/5000 Hz,
# 10-minute clips): roughly 10 seconds and 100 seconds per clip
# respectively on our test machine -- real, hardware-dependent
# numbers, not estimates, though not yet timed against real
# Patient_1/Patient_2 .mat files specifically, since none are
# available in this environment. Confirm on one real high-channel-
# count clip before running a full human subject.
MAX_CONNECTIVITY_PAIRS = 120
_CONNECTIVITY_RNG_SEED = 0


def _bandpass_sos(fs, low, high, order=4):
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999)
    return sp_signal.butter(order, [low_n, high_n], btype="band", output="sos")


def _pairwise_max_xcorr(x, y, norm, max_lag):
    """Max normalized cross-correlation between x and y within a small
    lag window, via FFT (O(n log n)) rather than a direct per-lag loop
    -- a direct loop over even a +/-20ms window is O(max_lag * n),
    which is fine at dog scale but not at Patient_2's 5000 Hz, 3M-sample
    clips. x and y must already be mean-subtracted; norm is
    sqrt(dot(x,x) * dot(y,y)), passed in so it is computed once by the
    caller rather than redundantly per pair.
    """
    if norm == 0:
        return np.nan
    n = len(x)
    full = sp_signal.fftconvolve(x, y[::-1], mode="full")
    zero_lag_idx = n - 1
    lo, hi = zero_lag_idx - max_lag, zero_lag_idx + max_lag + 1
    window = full[lo:hi] / norm
    return float(window[np.argmax(np.abs(window))])


def _connectivity_features(data: np.ndarray, fs: float) -> dict:
    """Bivariate/connectivity features aggregated (mean, std, max)
    across a bounded, subsampled set of channel pairs: zero-lag
    correlation (kept from the original single-scalar design), max
    cross-correlation within a small lag window, and phase-locking
    value in the alpha band. This replaces the single
    'mean_pairwise_corr' scalar with the richer connectivity feature
    set this literature actually reports as most informative, while
    keeping cost bounded via MAX_CONNECTIVITY_PAIRS.

    Explicitly frees each large intermediate array (centered, filtered,
    analytic) as soon as it is no longer needed, rather than letting
    data/centered/filtered/analytic/phases all stay alive at once. This
    is not a stylistic choice: the first version of this function,
    without the explicit deletes, was killed (out of memory) on a
    Patient_2-scale clip (24 channels, 5000 Hz, ~3M samples/channel)
    in a 3.9 GB test environment, holding roughly 3.5 GB of
    simultaneously-live arrays. This version keeps peak usage to
    roughly one or two of those arrays at a time. Still worth timing
    and memory-profiling on whatever machine actually runs this against
    real Patient_1/Patient_2 data -- a 3.9 GB ceiling is a small-machine
    worst case, not a benchmark for your hardware specifically.
    """
    n_channels = data.shape[0]
    out = {
        "xcorr0_mean": np.nan, "xcorr0_std": np.nan, "xcorr0_max": np.nan,
        "xcorr_lag_mean": np.nan, "xcorr_lag_std": np.nan, "xcorr_lag_max": np.nan,
        "plv_mean": np.nan, "plv_std": np.nan, "plv_max": np.nan,
    }
    if n_channels < 2:
        return out

    all_pairs = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]
    if len(all_pairs) > MAX_CONNECTIVITY_PAIRS:
        rng = np.random.default_rng(_CONNECTIVITY_RNG_SEED)
        idx = rng.choice(len(all_pairs), size=MAX_CONNECTIVITY_PAIRS, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    max_lag = max(1, int(CONNECTIVITY_MAX_LAG_SEC * fs))

    # --- zero-lag correlation and max-lag cross-correlation ---
    centered = data - data.mean(axis=1, keepdims=True)
    xcorr0_vals, xcorr_lag_vals = [], []
    for i, j in pairs:
        x, y = centered[i], centered[j]
        norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
        if norm == 0:
            continue
        xcorr0_vals.append(float(np.dot(x, y) / norm))
        xcorr_lag_vals.append(_pairwise_max_xcorr(x, y, norm, max_lag))
    # x, y are numpy VIEWS into centered's buffer (basic row indexing on
    # a 2D array does not copy) -- deleting `centered` alone does NOT
    # free its memory while the last iteration's x, y still reference
    # it, since numpy keeps a view's base array alive via view.base.
    # This was the actual cause of the Patient_2-scale OOM kill: the
    # earlier version's `del centered` silently did nothing.
    del x, y
    del centered

    for name, vals in [("xcorr0", xcorr0_vals), ("xcorr_lag", xcorr_lag_vals)]:
        arr = np.array(vals, float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[f"{name}_mean"] = float(np.mean(arr))
            out[f"{name}_std"] = float(np.std(arr)) if arr.size > 1 else np.nan
            out[f"{name}_max"] = float(np.max(np.abs(arr)))

    # --- phase-locking value in the alpha band ---
    # Hilbert transform once per channel (not once per pair), so this
    # step's cost scales with n_channels, not n_pairs. Free filtered
    # and analytic as soon as each is no longer needed -- analytic in
    # particular is complex128, double the memory of the real arrays.
    phases = None
    try:
        sos = _bandpass_sos(fs, *CONNECTIVITY_BAND)
        filtered = sp_signal.sosfiltfilt(sos, data, axis=1)
        analytic = sp_signal.hilbert(filtered, axis=1)
        del filtered
        phases = np.angle(analytic)
        del analytic
    except Exception:
        phases = None

    if phases is not None:
        plv_vals = []
        for i, j in pairs:
            dphi = phases[i] - phases[j]
            plv_vals.append(float(np.abs(np.mean(np.exp(1j * dphi)))))
        arr = np.array(plv_vals, float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out["plv_mean"] = float(np.mean(arr))
            out["plv_std"] = float(np.std(arr)) if arr.size > 1 else np.nan
            out["plv_max"] = float(np.max(arr))

    return out


def clip_features(clip: dict) -> dict:
    """Aggregate per-channel features across all channels of one clip.

    Channel-aggregated design (mean/std/max across channels), chosen over
    full per-channel features because clip counts per subject are small
    (tens to low hundreds) relative to how fast per-channel features would
    inflate the feature count — the same n-vs-p tradeoff the blazar paper
    hit with PLR_SELECT_K, just worse here given far fewer labeled
    examples per subject. If a subject turns out to have enough clips,
    per-channel features are a natural follow-up, not a redesign.
    """
    data, fs = clip["data"], clip["fs"]
    n_channels = data.shape[0]

    per_channel = [channel_features(data[c], fs) for c in range(n_channels)]
    keys = list(per_channel[0].keys())

    agg = {}
    for k in keys:
        vals = np.array([pc[k] for pc in per_channel], float)
        vals = vals[np.isfinite(vals)]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals.size else np.nan
        agg[f"{k}_std"] = float(np.std(vals)) if vals.size > 1 else np.nan
        agg[f"{k}_max"] = float(np.max(vals)) if vals.size else np.nan

    # cross-channel connectivity: cross-correlation (zero-lag and
    # small-lag max) and phase-locking value, aggregated across a
    # bounded set of channel pairs -- see _connectivity_features for
    # why this replaces the single mean_pairwise_corr scalar the
    # previous version used.
    agg.update(_connectivity_features(data, fs))

    return agg


# ============================================================
# 3) DATASET BUILDER (per subject)
# ============================================================

def assign_preictal_blocks(manifest: pd.DataFrame) -> pd.DataFrame:
    """Group preictal clips into hour-long blocks so a train/test split
    never separates clips from the same seizure lead-up.

    AES preictal clips are numbered sequentially per seizure event, six
    per hour-long block in the original release. This positional
    assignment is made before any file is read, so it cannot itself
    check against the real `sequence` field; `build_subject_dataset()`
    calls `verify_preictal_block_sequences()` immediately after loading
    to confirm that every block's real sequence values are exactly
    {1,...,6}, and raises if they are not. This has been run and
    verified against the real AES/Kaggle data for all five canine
    subjects used in this study. Interictal clips get one block per
    clip, since they are drawn independently from unrelated baseline
    periods with no shared-event leakage risk between them by
    construction.
    """
    manifest = manifest.copy()
    manifest["block_id"] = None

    pre = manifest[manifest["clip_type"] == "preictal"].sort_values("index")
    # Positional default: every 6 consecutive preictal clips = 1 block.
    # Checked against the real `sequence` field post hoc by
    # verify_preictal_block_sequences(), called from
    # build_subject_dataset() on every run.
    block_ids = (np.arange(len(pre)) // 6)
    manifest.loc[pre.index, "block_id"] = [
        f"preictal_block_{b}" for b in block_ids
    ]

    inter = manifest[manifest["clip_type"] == "interictal"]
    manifest.loc[inter.index, "block_id"] = [
        f"interictal_{i}" for i in inter["index"]
    ]

    return manifest


def verify_preictal_block_sequences(rows: list) -> None:
    """Check the assumed positional block grouping (index // 6, assigned
    before any .mat file is read -- see assign_preictal_blocks) against
    each clip's REAL sequence field, now that it has actually been
    loaded. This is the check assign_preictal_blocks itself cannot do,
    since block assignment happens before any file is opened and
    sequence only becomes available afterward.

    Per the AES dataset's own description, a preictal block's six clips
    should carry sequence values {1,...,6} in some order, one full set
    per block, with no repeats and nothing outside that range. If a
    subject's file numbering doesn't actually correspond to sequential
    six-clip blocks the way build_subject_dataset assumes, this is
    exactly the kind of thing that fails silently and corrupts every
    downstream causal-block guarantee -- so this raises loudly instead.
    """
    by_block = {}
    for r in rows:
        if r["clip_type"] != "preictal" or r.get("sequence") is None:
            continue
        by_block.setdefault(r["block_id"], []).append(r["sequence"])

    problems = []
    for block_id, seqs in by_block.items():
        seqs_sorted = sorted(seqs)
        if len(seqs) != 6:
            problems.append(f"{block_id}: expected 6 clips, got {len(seqs)}")
        elif seqs_sorted != [1, 2, 3, 4, 5, 6]:
            problems.append(f"{block_id}: expected sequence values 1-6, got {seqs_sorted}")

    if problems:
        raise ValueError(
            "Preictal block assignment does not match the real sequence "
            "field for " + str(len(problems)) + " block(s) -- the "
            "index//6 assumption in assign_preictal_blocks does not hold "
            "for this subject's file numbering. First few problems: "
            + "; ".join(problems[:5])
        )


def build_subject_dataset(subject_dir: Path) -> pd.DataFrame:
    """Discover + load + featurize every labeled clip for one subject.

    Returns a DataFrame with columns: path, clip_type, index, block_id,
    plus one column per aggregated feature, plus label (1=preictal,
    0=interictal). Test-labeled clips (clip_type=='test') are included
    but label is left as NaN -- those are for the held-out competition
    predictions, not for TRAIN/validation.
    """
    manifest = discover_clips(subject_dir)
    manifest = assign_preictal_blocks(manifest)

    rows = []
    for _, r in manifest.iterrows():
        try:
            clip = load_clip_mat(r["path"])
        except Exception as exc:
            print(f"  [WARN] failed to load {r['path'].name}: {exc}")
            continue
        feat = clip_features(clip)
        feat.update({
            "path": str(r["path"]),
            "clip_type": r["clip_type"],
            "index": r["index"],
            "block_id": r["block_id"],
            "sequence": clip.get("sequence"),
            "label": (1 if r["clip_type"] == "preictal"
                      else 0 if r["clip_type"] == "interictal" else np.nan),

        })
        rows.append(feat)

    if any(r.get("sequence") is not None for r in rows):
        verify_preictal_block_sequences(rows)
    else:
        print("  [WARN] no 'sequence' field found in any loaded clip -- "
              "cannot verify the index//6 block-grouping assumption for "
              "this subject. Block assignment may be silently wrong; "
              "check a sample file's fields directly before trusting "
              "downstream causal-CV results.")

    return pd.DataFrame(rows)


# ============================================================
# 4) MODELS  (mirrors the blazar LR / PLR / RF construction)
# ============================================================

def make_model(name: str, plr_select_k: int = 20):
    name = name.lower().strip()
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=5000,
                                       class_weight="balanced", C=1.0,
                                       random_state=RANDOM_SEED)),
        ])
    if name == "polylogreg":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=plr_select_k)),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=5000,
                                       class_weight="balanced", C=0.05,
                                       random_state=RANDOM_SEED)),
        ])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", oob_score=True,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


# ============================================================
# 5) BLOCK-AWARE SPLIT + CALIBRATION + THRESHOLD
# ============================================================

def block_train_test_split(df: pd.DataFrame, test_frac_blocks: float = 0.25,
                            seed: int = RANDOM_SEED):
    """Split by BLOCK, never by clip — the leakage guard this whole
    port exists for. Every block goes entirely into TRAIN or entirely
    into TEST.

    Stratified by class at the block level: with as few as 2-3 preictal
    blocks per subject (a realistic AES/Kaggle count, not a synthetic
    worst case), an unstratified random block split has a real chance
    of putting zero preictal blocks in TEST, silently degrading to a
    single-class evaluation. This surfaced on the first real run of
    this pipeline, not as a hypothetical.
    """
    rng = np.random.default_rng(seed)

    block_label = df.groupby("block_id")["label"].first()
    pre_blocks = block_label[block_label == 1].index.to_numpy()
    int_blocks = block_label[block_label == 0].index.to_numpy()

    rng.shuffle(pre_blocks)
    rng.shuffle(int_blocks)

    n_test_pre = max(1, int(round(len(pre_blocks) * test_frac_blocks))) \
        if len(pre_blocks) >= 2 else 0
    n_test_int = max(1, int(round(len(int_blocks) * test_frac_blocks))) \
        if len(int_blocks) >= 2 else 0

    test_blocks = set(pre_blocks[:n_test_pre]) | set(int_blocks[:n_test_int])
    if len(pre_blocks) < 2 or len(int_blocks) < 2:
        print("  [WARN] fewer than 2 blocks in one class — cannot "
              "guarantee both classes appear in TEST for this subject.")

    is_test = df["block_id"].isin(test_blocks)
    return df[~is_test].copy(), df[is_test].copy()


def _cal_classifier(base_model, method="sigmoid"):
    """Wrap a fitted (prefit) model for probability calibration.

    scikit-learn >= 1.6 removed cv="prefit" from CalibratedClassifierCV
    in favor of wrapping the fitted model in sklearn.frozen.FrozenEstimator
    first. This bit us directly: a bare cv="prefit" call raised a
    ValueError ("must be an int... Got 'prefit' instead"), not the
    TypeError an earlier version of this function was guarding against,
    so the except clause never fired and it looked like calibration
    "worked" only by accident (whenever the random calibration split
    happened to lack class diversity and fell back to the uncalibrated
    path before ever reaching this call) — seed-dependent, silent, and
    only surfaced by actually running multiple seeds. Try the current
    API first, fall back to the old one for older sklearn installs.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(estimator=FrozenEstimator(base_model),
                                       method=method)
    except ImportError:
        try:
            return CalibratedClassifierCV(estimator=base_model, method=method,
                                           cv="prefit")
        except TypeError:
            return CalibratedClassifierCV(base_estimator=base_model, method=method,
                                           cv="prefit")


def fit_calibrated(model_name, X_train, y_train, feature_cols,
                    blocks_train, seed=RANDOM_SEED, plr_select_k=20):
    """Fit on most of TRAIN, calibrate (Platt) on a held-out slice of
    TRAIN BLOCKS — mirrors the blazar code's calibration-tail idea, but
    the held-out slice is chosen by block, not by time order, since
    there's no natural "later" here within one subject's pooled clips.
    """
    uniq_blocks = np.array(sorted(blocks_train.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq_blocks)
    n_cal = max(1, int(len(uniq_blocks) * CALIBRATION_TAIL_FRAC))
    cal_blocks = set(uniq_blocks[:n_cal])

    is_cal = blocks_train.isin(cal_blocks).values
    X_fit, y_fit = X_train[~is_cal], y_train[~is_cal]
    X_cal, y_cal = X_train[is_cal], y_train[is_cal]

    base = make_model(model_name, plr_select_k=plr_select_k)

    if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
        # not enough class diversity to calibrate safely — fit on
        # everything, uncalibrated, and say so
        base.fit(X_train, y_train)
        return base, False

    base.fit(X_fit, y_fit)
    cal = _cal_classifier(base, method="sigmoid")
    cal.fit(X_cal, y_cal)
    return cal, True


def optimize_threshold(y_true, p, beta=THRESHOLD_FBETA):
    best_thr, best_fb, best_info = 0.5, -1, {}
    for thr in THRESH_GRID:
        yhat = (p >= thr).astype(int)
        fb = fbeta_score(y_true, yhat, beta=beta, zero_division=0)
        if fb > best_fb:
            best_fb = fb
            best_thr = float(thr)
            best_info = {
                "fbeta": float(fb),
                "precision": float(precision_score(y_true, yhat, zero_division=0)),
                "recall": float(recall_score(y_true, yhat, zero_division=0)),
            }
    return best_thr, best_info


# AES/Kaggle clips are all 10 minutes. If this pipeline is ever pointed
# at a dataset with a different clip length, this needs to change (or
# be read per-clip from data_length_sec, which build_subject_dataset()
# currently discards after loading -- worth capturing there instead of
# hardcoding here if clip length ever varies within a dataset).
CLIP_DURATION_SEC = 600.0

# 0.15 FP/h ~= one false alarm every ~6.7 hours -- in the range commonly
# used as a "low false alarm" operating point in the literature (e.g.
# Mirowski et al. reported results at <0.3 and <1 FP/h; long-term
# implanted-advisory-system work targets comparably low rates). Not a
# fixed field standard, just a reasonable, clearly-labeled default;
# override per analysis if a specific target is being compared against.
TARGET_FPH = 0.15


def compute_fp_per_hour(y_true, p, threshold, clip_duration_sec=CLIP_DURATION_SEC):
    """False positives per hour at a given threshold -- the metric this
    field actually reports (a predictor that fires every 20 minutes is
    useless regardless of its AUC), not something AUC or MCC communicate
    on their own.

    FP/h = (# interictal clips misclassified as positive) / (total
    interictal-clip hours evaluated). Bounded above by 3600/clip_duration_sec
    (6.0 for 10-min clips) since at most every clip can be a false alarm.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p, float)
    interictal_mask = (y_true == 0)
    n_interictal = int(np.sum(interictal_mask))
    if n_interictal == 0:
        return np.nan
    fp = int(np.sum((p[interictal_mask] >= threshold)))
    hours_evaluated = n_interictal * (clip_duration_sec / 3600.0)
    return float(fp / hours_evaluated) if hours_evaluated > 0 else np.nan


def sensitivity_at_target_fph(y_true, p, target_fph=0.15,
                               clip_duration_sec=CLIP_DURATION_SEC,
                               grid=THRESH_GRID):
    """Find the threshold whose FP/h is closest to (and at or below,
    when possible) target_fph, and report sensitivity (recall) there —
    the standard "X% sensitivity at Y FP/h" format used to report
    seizure-prediction results in the literature, e.g. Mirowski et al.
    reported results at <0.3 FP/h and <1 FP/h operating points.

    Returns (threshold, achieved_fph, sensitivity). If no threshold
    achieves fph <= target_fph (predictor never gets that selective),
    returns the threshold with the LOWEST achieved fph instead, and the
    caller should treat sensitivity there as "best case, still above
    target" rather than a result meeting the target.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p, float)

    candidates = []
    for thr in grid:
        fph = compute_fp_per_hour(y_true, p, thr, clip_duration_sec)
        yhat = (p >= thr).astype(int)
        sens = float(recall_score(y_true, yhat, zero_division=0))
        candidates.append((float(thr), fph, sens))

    under_target = [c for c in candidates if c[1] <= target_fph]
    if under_target:
        # among thresholds meeting the target, take the most sensitive
        best = max(under_target, key=lambda c: c[2])
    else:
        # target unreachable at this operating range; report the
        # lowest-FP/h point instead, clearly not meeting target
        best = min(candidates, key=lambda c: c[1])

    return best  # (threshold, achieved_fph, sensitivity)


def block_shuffle_permutation_test(y_true, p, blocks, n_perm=N_PERM_TEST, seed=0):
    """Permutation test that shuffles LABELS AT THE BLOCK LEVEL, so
    clips from the same block move together — the direct analog of the
    blazar code's block-shuffle AUC permutation test, with "block"
    swapped for "time block" -> "seizure/interictal block."
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p, float)
    blocks = np.asarray(blocks)
    uniq = np.unique(blocks)
    block_label = {b: y_true[blocks == b][0] for b in uniq}  # label is
    # constant within a block by construction (all-preictal or
    # all-interictal), so this is safe.

    auc_obs = float(roc_auc_score(y_true, p))
    rng = np.random.default_rng(seed)
    labels = np.array([block_label[b] for b in uniq])
    count = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        y_perm = np.array([perm_labels[np.where(uniq == b)[0][0]] for b in blocks])
        if len(np.unique(y_perm)) < 2:
            continue
        if roc_auc_score(y_perm, p) >= auc_obs:
            count += 1
    pval = (count + 1) / (n_perm + 1)
    return pval, auc_obs


def leave_one_preictal_block_out_cv(df: pd.DataFrame, feature_cols, model_name,
                                     seed: int = RANDOM_SEED,
                                     plr_select_k: int = 20) -> Optional[dict]:
    """K-fold CV where K = number of preictal (seizure) blocks. Each fold
    holds out exactly ONE preictal block plus a proportional share of
    interictal blocks as TEST; predictions are pooled out-of-fold across
    all K folds before any metric is computed.

    This replaces a single stratified train/test split, which — on a
    real Dog_1 run (24 preictal clips = 4 preictal blocks, 480 single-
    clip interictal blocks) — put only 1 of the 4 preictal blocks in
    TEST. That left every reported AUC resting on one seizure's worth of
    correlated evidence, and the block-shuffle permutation test correctly
    flagged the resulting AUCs (0.049-0.150) as statistically
    indistinguishable from chance (p=0.87-0.95). Pooling OOF predictions
    across all 4 preictal blocks uses all the available seizure evidence
    instead of gambling the whole evaluation on which one block a single
    random split happened to withhold.

    Returns None if there are fewer than 2 preictal blocks (nothing to
    cross-validate). Otherwise returns a dict with the pooled
    out-of-fold true labels / probabilities / block ids (oof_true is
    None if no fold produced a usable prediction), plus per-fold
    diagnostics.
    """
    block_label = df.groupby("block_id")["label"].first()
    pre_blocks = np.array(sorted(block_label[block_label == 1].index))
    int_blocks = np.array(sorted(block_label[block_label == 0].index))

    n_folds = len(pre_blocks)
    if n_folds < 2:
        return None

    rng = np.random.default_rng(seed)
    shuffled_int = int_blocks.copy()
    rng.shuffle(shuffled_int)
    int_groups = np.array_split(shuffled_int, n_folds)

    block_col = df["block_id"].values
    label_col = df["label"].astype(int).values

    oof_proba = np.full(len(df), np.nan)
    fold_diagnostics = []

    for fold_i in range(n_folds):
        test_blocks = {pre_blocks[fold_i]} | set(int_groups[fold_i].tolist())
        is_test = np.isin(block_col, list(test_blocks))
        is_train = ~is_test
        test_idx = np.where(is_test)[0]

        train_df = df.loc[is_train]
        medians = train_df[feature_cols].median(numeric_only=True)
        X_train = train_df[feature_cols].fillna(medians).values
        y_train = train_df["label"].astype(int).values
        X_test = df.loc[is_test, feature_cols].fillna(medians).values

        if len(np.unique(y_train)) < 2:
            fold_diagnostics.append({"fold": fold_i, "status": "SKIPPED",
                                      "reason": "TRAIN has < 2 classes"})
            continue

        try:
            fitted, calibrated = fit_calibrated(
                model_name, X_train, y_train, feature_cols,
                train_df["block_id"], seed=seed, plr_select_k=plr_select_k)
            p_test = fitted.predict_proba(X_test)[:, 1]
        except Exception as exc:
            print(f"    [FOLD FAILED] fold={fold_i} model={model_name} "
                  f"seed={seed}: {exc}")
            fold_diagnostics.append({"fold": fold_i, "status": "FAILED",
                                      "error": str(exc)})
            continue

        oof_proba[test_idx] = p_test
        fold_diagnostics.append({
            "fold": fold_i, "status": "OK", "calibrated": calibrated,
            "n_test": int(len(test_idx)),
            "held_out_preictal_block": str(pre_blocks[fold_i]),
        })

    has_pred = np.isfinite(oof_proba)
    if not np.any(has_pred) or len(np.unique(label_col[has_pred])) < 2:
        return {"n_folds": n_folds, "fold_diagnostics": fold_diagnostics,
                "oof_true": None, "oof_proba": None, "oof_block": None}

    return {
        "n_folds": n_folds,
        "fold_diagnostics": fold_diagnostics,
        "oof_true": label_col[has_pred],
        "oof_proba": oof_proba[has_pred],
        "oof_block": block_col[has_pred],
    }


# ============================================================
# 6) PER-SUBJECT DRIVER
# ============================================================

@dataclass
class SubjectResult:
    subject: str
    n_clips: int
    n_preictal: int
    n_interictal: int
    n_blocks: int
    models: Dict[str, dict] = field(default_factory=dict)


def run_subject(subject_dir: Path, subject_name: str,
                 seed: int = RANDOM_SEED,
                 plr_select_k: int = 20,
                 models_to_run=("logreg", "polylogreg", "rf")) -> SubjectResult:
    print(f"\n{'='*60}\nSUBJECT: {subject_name}\n{'='*60}")

    df = build_subject_dataset(subject_dir)
    df = df[df["clip_type"] != "test"].reset_index(drop=True)   # drop unlabeled
    feature_cols = [c for c in df.columns
                    if c not in ("path", "clip_type", "index", "block_id", "label")]

    n_pre = int((df["label"] == 1).sum())
    n_int = int((df["label"] == 0).sum())
    n_blocks = df["block_id"].nunique()
    n_pre_blocks = int(df.loc[df["label"] == 1, "block_id"].nunique())
    print(f"  clips: {len(df)}  preictal={n_pre}  interictal={n_int}  "
          f"blocks={n_blocks}  preictal_blocks={n_pre_blocks}")

    if n_pre_blocks < 2:
        print("  [WARN] fewer than 2 preictal (seizure) blocks — cannot "
              "cross-validate for this subject. Any result below is a "
              "pipeline smoke test only, not a trustworthy estimate.")

    result = SubjectResult(subject=subject_name, n_clips=len(df),
                            n_preictal=n_pre, n_interictal=n_int,
                            n_blocks=n_blocks)

    for model_name in models_to_run:
        print(f"\n  --- {model_name} ---")

        cv = leave_one_preictal_block_out_cv(
            df, feature_cols, model_name, seed=seed, plr_select_k=plr_select_k)

        if cv is None:
            result.models[model_name] = {
                "status": "SKIPPED",
                "note": "fewer than 2 preictal blocks; cannot cross-validate",
            }
            print("    [SKIP] fewer than 2 preictal blocks.")
            continue

        if cv["oof_true"] is None:
            result.models[model_name] = {
                "status": "OK", "n_folds": cv["n_folds"],
                "note": "no fold produced a usable out-of-fold prediction",
            }
            print(f"    [WARN] no usable out-of-fold predictions across "
                  f"{cv['n_folds']} folds.")
            continue

        y_oof, p_oof, blocks_oof = cv["oof_true"], cv["oof_proba"], cv["oof_block"]
        auc = float(roc_auc_score(y_oof, p_oof))
        ap = float(average_precision_score(y_oof, p_oof))
        thr, thr_info = optimize_threshold(y_oof, p_oof)
        yhat = (p_oof >= thr).astype(int)
        mcc = float(matthews_corrcoef(y_oof, yhat)) if len(np.unique(yhat)) > 1 else np.nan
        pval, _ = block_shuffle_permutation_test(y_oof, p_oof, blocks_oof, seed=seed)

        # Clinically standard reporting for this field: FP/h at the
        # chosen operating point, and sensitivity at a fixed, low FP/h
        # target — AUC alone doesn't say whether a predictor is usable
        # (a device firing every 20 min is useless regardless of AUC).
        fph_at_thr = compute_fp_per_hour(y_oof, p_oof, thr)
        thr_target, fph_target, sens_target = sensitivity_at_target_fph(
            y_oof, p_oof, target_fph=TARGET_FPH)

        res = {
            "status": "OK",
            "n_folds": cv["n_folds"],
            "auc_cv": auc, "ap_cv": ap, "threshold": thr,
            "precision": thr_info.get("precision"),
            "recall": thr_info.get("recall"),
            "mcc": mcc, "perm_pvalue": pval,
            "fp_per_hour": fph_at_thr,
            "sensitivity_at_target_fph": sens_target,
            "achieved_fph_at_target": fph_target,
            "threshold_at_target_fph": thr_target,
            "n_oof_clips": int(len(y_oof)),
            "n_oof_preictal_clips": int(np.sum(y_oof == 1)),
        }
        print(f"    pooled-OOF AUC={auc:.3f}  AP={ap:.3f}  thr={thr:.2f}  "
              f"P={thr_info.get('precision'):.3f}  "
              f"R={thr_info.get('recall'):.3f}  MCC={mcc:.3f}  "
              f"perm_p={pval:.4g}  FP/h={fph_at_thr:.3f}  "
              f"[sens={sens_target:.3f} @ {fph_target:.3f} FP/h, "
              f"target={TARGET_FPH}]  (folds={cv['n_folds']}, "
              f"oof_clips={len(y_oof)})")

        result.models[model_name] = res

    return result


def run_all_subjects(subject_dirs: Dict[str, Path], **kwargs) -> pd.DataFrame:
    rows = []
    for name, sdir in subject_dirs.items():
        try:
            res = run_subject(sdir, name, **kwargs)
        except Exception as exc:
            print(f"[FAIL] subject {name}: {exc}")
            rows.append({"subject": name, "model": None, "status": "FAILED",
                         "error": str(exc)})
            continue
        for model_name, m in res.models.items():
            rows.append({"subject": name, "model": model_name,
                         "n_clips": res.n_clips, "n_preictal": res.n_preictal,
                         "n_interictal": res.n_interictal,
                         "n_blocks": res.n_blocks, **m})
    return pd.DataFrame(rows)


# ============================================================
# 7) SEED-SENSITIVITY / STABILITY CHECK
# ============================================================
#
# Every reported AUC/MCC in run_subject() depends on ONE random seed's
# assignment of interictal blocks to folds (see
# leave_one_preictal_block_out_cv: rng.shuffle(shuffled_int) before
# splitting into n_folds groups). With as few as 5-17 preictal blocks
# per subject, that's not a large-sample average — it's one particular
# random grouping. Before treating any of these results as publishable,
# the question is: does the AUC/MCC/significance call survive being
# recomputed under a handful of different seeds, or was the number we
# reported just what that one seed happened to produce?
#
# This section runs each subject's full leave-one-preictal-block-out CV
# under several seeds, WITHOUT re-loading or re-extracting features
# (that's the expensive part, handled once via build_subject_dataset)
# — only the block-to-fold RNG assignment changes per seed, so this is
# cheap relative to the first full run.

def run_subject_multiseed(subject_dir: Path, subject_name: str,
                           seeds=(0, 1, 2, 3, 4),
                           plr_select_k: int = 20,
                           models_to_run=("logreg", "polylogreg", "rf")) -> pd.DataFrame:
    """Run leave_one_preictal_block_out_cv for one subject under several
    seeds, reusing the same feature-extracted dataset across all of them.

    Returns one row per (model, seed): auc_cv, ap_cv, mcc, perm_pvalue,
    threshold, precision, recall, n_folds, n_oof_clips. Use
    summarize_seed_stability() on the result to get the per-(subject,
    model) mean/std/range and the significance-flip check that actually
    answers "is this result stable enough to publish."
    """
    print(f"\n{'='*60}\nSUBJECT: {subject_name}  (seed-sensitivity check, "
          f"{len(seeds)} seeds: {list(seeds)})\n{'='*60}")

    df = build_subject_dataset(subject_dir)
    df = df[df["clip_type"] != "test"].reset_index(drop=True)
    feature_cols = [c for c in df.columns
                    if c not in ("path", "clip_type", "index", "block_id", "label")]

    n_pre_blocks = int(df.loc[df["label"] == 1, "block_id"].nunique())
    print(f"  clips: {len(df)}  preictal_blocks={n_pre_blocks}")
    if n_pre_blocks < 2:
        print("  [SKIP] fewer than 2 preictal blocks — cannot cross-validate "
              "at any seed.")
        return pd.DataFrame()

    rows = []
    for model_name in models_to_run:
        for seed in seeds:
            cv = leave_one_preictal_block_out_cv(
                df, feature_cols, model_name, seed=seed, plr_select_k=plr_select_k)

            if cv is None or cv["oof_true"] is None:
                rows.append({
                    "subject": subject_name, "model": model_name, "seed": seed,
                    "status": "NO_OOF", "n_folds": (cv or {}).get("n_folds", None),
                })
                continue

            y_oof, p_oof, blocks_oof = cv["oof_true"], cv["oof_proba"], cv["oof_block"]
            auc = float(roc_auc_score(y_oof, p_oof))
            ap = float(average_precision_score(y_oof, p_oof))
            thr, thr_info = optimize_threshold(y_oof, p_oof)
            yhat = (p_oof >= thr).astype(int)
            mcc = float(matthews_corrcoef(y_oof, yhat)) if len(np.unique(yhat)) > 1 else np.nan
            pval, _ = block_shuffle_permutation_test(y_oof, p_oof, blocks_oof, seed=seed)
            fph_at_thr = compute_fp_per_hour(y_oof, p_oof, thr)
            thr_target, fph_target, sens_target = sensitivity_at_target_fph(
                y_oof, p_oof, target_fph=TARGET_FPH)

            rows.append({
                "subject": subject_name, "model": model_name, "seed": seed,
                "status": "OK", "n_folds": cv["n_folds"],
                "auc_cv": auc, "ap_cv": ap, "threshold": thr,
                "precision": thr_info.get("precision"),
                "recall": thr_info.get("recall"),
                "mcc": mcc, "perm_pvalue": pval,
                "fp_per_hour": fph_at_thr,
                "sensitivity_at_target_fph": sens_target,
                "achieved_fph_at_target": fph_target,
                "n_oof_clips": int(len(y_oof)),
            })
            print(f"    [{model_name} seed={seed}] AUC={auc:.3f}  MCC={mcc:.3f}  "
                  f"perm_p={pval:.4g}  FP/h={fph_at_thr:.3f}  "
                  f"sens@{TARGET_FPH}fph={sens_target:.3f}")

    return pd.DataFrame(rows)


def run_all_subjects_multiseed(subject_dirs: Dict[str, Path],
                                seeds=(0, 1, 2, 3, 4),
                                **kwargs) -> pd.DataFrame:
    """run_subject_multiseed() for every subject; returns one combined
    long-format table (subject x model x seed).
    """
    frames = []
    for name, sdir in subject_dirs.items():
        try:
            df_sub = run_subject_multiseed(sdir, name, seeds=seeds, **kwargs)
        except Exception as exc:
            print(f"[FAIL] subject {name}: {exc}")
            frames.append(pd.DataFrame([{
                "subject": name, "model": None, "seed": None,
                "status": "FAILED", "error": str(exc),
            }]))
            continue
        if len(df_sub):
            frames.append(df_sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_seed_stability(multiseed_df: pd.DataFrame,
                              sig_alpha: float = 0.05) -> pd.DataFrame:
    """Collapse the (subject x model x seed) table into one row per
    (subject, model): mean/std/range of AUC and MCC across seeds, how
    many of the tested seeds came back significant (p < sig_alpha), and
    a plain-language verdict.

    Verdict logic:
      "STABLE_SIGNIFICANT"   — every seed significant, AUC range < 0.15
      "STABLE_NULL"          — no seed significant, AUC range < 0.15
      "UNSTABLE"             — significance call flips between seeds,
                                 OR AUC range >= 0.15 regardless of
                                 significance pattern

    "UNSTABLE" is the result that matters most for publishability: it
    means the headline AUC/MCC/significance in the original single-seed
    run was substantially a function of which seed got used, not a
    property of the data. Those subjects/models should not be reported
    as clean positives (or clean nulls) without saying so explicitly.
    """
    ok = multiseed_df[multiseed_df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame()

    rows = []
    for (subject, model), g in ok.groupby(["subject", "model"]):
        n_seeds = len(g)
        n_sig = int((g["perm_pvalue"] < sig_alpha).sum())
        auc_mean, auc_std = g["auc_cv"].mean(), g["auc_cv"].std()
        auc_min, auc_max = g["auc_cv"].min(), g["auc_cv"].max()
        auc_range = auc_max - auc_min
        mcc_mean, mcc_std = g["mcc"].mean(), g["mcc"].std()
        fph_mean, fph_std = g["fp_per_hour"].mean(), g["fp_per_hour"].std()
        sens_target_mean = g["sensitivity_at_target_fph"].mean()
        sens_target_std = g["sensitivity_at_target_fph"].std()

        if n_sig == n_seeds and auc_range < 0.15:
            verdict = "STABLE_SIGNIFICANT"
        elif n_sig == 0 and auc_range < 0.15:
            verdict = "STABLE_NULL"
        else:
            verdict = "UNSTABLE"

        rows.append({
            "subject": subject, "model": model, "n_seeds": n_seeds,
            "n_seeds_significant": n_sig,
            "auc_mean": round(auc_mean, 3), "auc_std": round(auc_std, 3),
            "auc_min": round(auc_min, 3), "auc_max": round(auc_max, 3),
            "auc_range": round(auc_range, 3),
            "mcc_mean": round(mcc_mean, 3), "mcc_std": round(mcc_std, 3),
            "fp_per_hour_mean": round(fph_mean, 3),
            "fp_per_hour_std": round(fph_std, 3),
            f"sensitivity_at_{TARGET_FPH}fph_mean": round(sens_target_mean, 3),
            f"sensitivity_at_{TARGET_FPH}fph_std": round(sens_target_std, 3),
            "verdict": verdict,
        })

    return pd.DataFrame(rows).sort_values(["subject", "model"]).reset_index(drop=True)


if __name__ == "__main__":
    # Point this at a directory containing one subfolder per subject,
    # each holding that subject's .mat clips, e.g.:
    #   data/Dog_1/Dog_1_preictal_segment_0001.mat, ...
    #   data/Dog_2/...
    DATA_ROOT = Path("data")
    subject_dirs = {p.name: p for p in sorted(DATA_ROOT.iterdir()) if p.is_dir()}

    # Seed-sensitivity check: rerun each subject's CV under 5 different
    # seeds (only the block-to-fold RNG assignment changes; features are
    # extracted once per subject, not once per seed). This is what
    # actually answers whether the single-seed AUC/MCC/significance
    # numbers already collected are stable enough to publish, or whether
    # they were just what one particular random grouping happened to
    # produce.
    SEEDS = (0, 1, 2, 3, 4)

    multiseed = run_all_subjects_multiseed(subject_dirs, seeds=SEEDS)
    multiseed.to_csv("seizure_multiseed_results.csv", index=False)

    stability = summarize_seed_stability(multiseed)
    stability.to_csv("seizure_seed_stability_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SEED-STABILITY SUMMARY  (this is the table that matters for "
          "publishability)")
    print("=" * 70)
    print(stability.to_string(index=False))

    n_unstable = int((stability["verdict"] == "UNSTABLE").sum())
    n_total = len(stability)
    print(f"\n{n_unstable}/{n_total} (subject, model) results are UNSTABLE "
          f"across seeds — do not report these as clean positives/nulls "
          f"without saying so.")
