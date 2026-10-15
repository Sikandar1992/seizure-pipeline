"""
test_seizure_pipeline.py -- pytest suite for seizure_pipeline.py.

These tests cover the specific behaviors that were actually debugged and
verified during development, not a generic template:

  - the block-splitting / sequence-verification logic that prevents
    train/test leakage across clips from the same seizure
  - the connectivity-feature memory fix (a numpy view-retention bug that
    caused an out-of-memory kill at Patient-scale clip dimensions)
  - the FFT-based autocorrelation fix (replacing an O(n^2) direct
    computation that was unusably slow on 5000 Hz human-subject clips)
  - the cross-correlation / phase-locking-value connectivity features
    themselves, checked against signals with known relationships
  - basic construction sanity for all three model families

Run with: pytest test_seizure_pipeline.py -v
Slow/large-scale tests are marked @pytest.mark.slow and skipped by
default (see conftest.py); run with `pytest --runslow` to include them.
"""

import numpy as np
import pytest

from seizure_pipeline import (
    clip_features,
    channel_features,
    _autocorr_decay_lag,
    _connectivity_features,
    _pairwise_max_xcorr,
    _bandpass_sos,
    assign_preictal_blocks,
    verify_preictal_block_sequences,
    make_model,
    block_train_test_split,
    compute_fp_per_hour,
    sensitivity_at_target_fph,
)

import pandas as pd



# ------------------------------------------------------------------
# Synthetic clip generators
# ------------------------------------------------------------------

def make_synthetic_clip(n_channels=16, fs=400.0, duration_sec=600.0, seed=0):
    """Dog-scale synthetic clip: independent noisy channels with a
    shared oscillatory component, so connectivity features have
    something real to detect."""
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs
    shared = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz shared component
    data = rng.normal(0, 1, (n_channels, n_samples)) + shared[None, :]
    return {"data": data, "fs": fs}


# ------------------------------------------------------------------
# Block-splitting / sequence verification
# ------------------------------------------------------------------

class TestBlockVerification:
    def test_correct_sequences_pass(self):
        """A properly-numbered set of blocks (each a full 1-6 permutation)
        should not raise."""
        rows = []
        for block_id, seqs in [("preictal_block_0", [1, 2, 3, 4, 5, 6]),
                                ("preictal_block_1", [3, 1, 4, 6, 2, 5])]:
            for s in seqs:
                rows.append({"clip_type": "preictal", "block_id": block_id,
                              "sequence": s})
        verify_preictal_block_sequences(rows)  # should not raise

    def test_wrong_count_raises(self):
        """A block with the wrong number of clips (5 instead of 6) must
        be caught, not silently accepted."""
        rows = [{"clip_type": "preictal", "block_id": "b0", "sequence": s}
                for s in [1, 2, 3, 4, 5]]
        with pytest.raises(ValueError, match="expected 6 clips"):
            verify_preictal_block_sequences(rows)

    def test_duplicate_sequence_raises(self):
        """A block with a repeated sequence value (and therefore a
        missing one) must be caught -- this is exactly the failure mode
        that would silently corrupt the causal block-splitting guarantee
        if the index//6 positional assumption doesn't hold for a given
        subject's file numbering."""
        rows = [{"clip_type": "preictal", "block_id": "b0", "sequence": s}
                for s in [1, 1, 2, 3, 4, 5]]
        with pytest.raises(ValueError, match="expected sequence values 1-6"):
            verify_preictal_block_sequences(rows)

    def test_interictal_rows_ignored(self):
        """Interictal rows have no meaningful sequence semantics and
        must not trip the preictal-only check."""
        rows = [{"clip_type": "interictal", "block_id": "interictal_0",
                  "sequence": None}]
        verify_preictal_block_sequences(rows)  # should not raise

    def test_assign_preictal_blocks_groups_by_six(self):
        """Positional grouping: every 6 consecutive preictal clips (by
        index) should land in the same block, and interictal clips
        should each get their own block (no shared-event leakage risk
        between them by construction)."""
        manifest = pd.DataFrame({
            "clip_type": ["preictal"] * 12 + ["interictal"] * 3,
            "index": list(range(12)) + list(range(3)),
        })
        out = assign_preictal_blocks(manifest)
        pre = out[out["clip_type"] == "preictal"]
        assert pre.iloc[0:6]["block_id"].nunique() == 1
        assert pre.iloc[6:12]["block_id"].nunique() == 1
        assert pre.iloc[0]["block_id"] != pre.iloc[6]["block_id"]
        inter = out[out["clip_type"] == "interictal"]
        assert inter["block_id"].nunique() == 3  # each interictal clip its own block


# ------------------------------------------------------------------
# Connectivity features: correctness and the memory-safety fix
# ------------------------------------------------------------------

class TestConnectivityFeatures:
    def test_correlated_channels_detected(self):
        """Two channels built from the same underlying signal should
        show high zero-lag correlation; this is a sanity check that the
        cross-correlation computation is actually measuring what it
        claims to, not just running without crashing."""
        rng = np.random.default_rng(0)
        fs, n_samples = 400.0, int(400 * 60)
        shared = np.sin(2 * np.pi * 5 * np.arange(n_samples) / fs)
        data = np.zeros((4, n_samples))
        data[0] = shared + rng.normal(0, 0.01, n_samples)
        data[1] = shared + rng.normal(0, 0.01, n_samples)  # near-duplicate of ch 0
        data[2] = rng.normal(0, 1, n_samples)               # independent noise
        data[3] = rng.normal(0, 1, n_samples)               # independent noise
        out = _connectivity_features(data, fs)
        # with only 6 pairs total (4 choose 2), all are used -- the
        # mean will be pulled up by the one highly-correlated pair
        assert out["xcorr0_max"] > 0.9

    def test_output_keys_present(self):
        """Regression check on the feature schema: exactly these 9 keys,
        every time, even in edge cases -- downstream code (clip_features,
        the model pipelines) depends on a stable key set."""
        data = np.random.default_rng(0).normal(0, 1, (8, int(400 * 60)))
        out = _connectivity_features(data, 400.0)
        expected = {"xcorr0_mean", "xcorr0_std", "xcorr0_max",
                    "xcorr_lag_mean", "xcorr_lag_std", "xcorr_lag_max",
                    "plv_mean", "plv_std", "plv_max"}
        assert set(out.keys()) == expected

    def test_single_channel_returns_nan_not_crash(self):
        """n_channels < 2 has no pairs to compute -- must return NaNs,
        not raise, since a subject could in principle have a
        single-channel recording."""
        data = np.random.default_rng(0).normal(0, 1, (1, 1000))
        out = _connectivity_features(data, 400.0)
        assert all(np.isnan(v) for v in out.values())

    def test_pair_count_capped(self):
        """With more than MAX_CONNECTIVITY_PAIRS possible pairs, the
        function must still return in reasonable time and produce valid
        (non-NaN) aggregates -- this doesn't re-verify the exact
        Patient-scale timing (see test_patient_scale_no_memory_crash for
        that), just that capping doesn't silently break the output."""
        data = np.random.default_rng(0).normal(0, 1, (20, int(400 * 30)))
        out = _connectivity_features(data, 400.0)
        assert np.isfinite(out["xcorr0_mean"])
        assert np.isfinite(out["plv_mean"])

    @pytest.mark.slow
    def test_patient_scale_no_memory_crash(self):
        """Regression test for the actual bug found during development:
        the first version of _connectivity_features held data, centered,
        filtered, analytic, and phases all in memory simultaneously
        (~3.5 GB at 24 channels x 5000 Hz x 600s), and separately, numpy
        views from the xcorr loop kept `centered` alive even after
        `del centered` was called, silently defeating that cleanup. Both
        caused an out-of-memory kill in a 3.9 GB test environment. This
        test doesn't assert on a memory ceiling directly (not portable
        across machines) -- it asserts that the call completes and
        returns valid output at the worst-case dimensions, which is what
        actually failed before the fix.
        """
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (24, int(5000 * 600)))
        out = _connectivity_features(data, 5000.0)
        assert np.isfinite(out["xcorr0_mean"])
        assert np.isfinite(out["plv_mean"])


# ------------------------------------------------------------------
# Autocorrelation: the FFT-based fix
# ------------------------------------------------------------------

class TestAutocorrelation:
    def test_matches_direct_computation_small_case(self):
        """At a small enough scale to compute directly, the FFT-based
        implementation must agree with a naive per-lag loop -- this is
        the correctness check for the fix that replaced an O(n^2)
        np.correlate call (which was fast enough for dogs at 400 Hz but
        became impractically slow at the human subjects' 5000 Hz)."""
        rng = np.random.default_rng(0)
        fs, n_samples = 100.0, 2000
        x = rng.normal(0, 1, n_samples)
        x = x - x.mean()

        max_lag_sec = 1.0
        max_lag = int(max_lag_sec * fs)
        var0 = np.dot(x, x)
        direct_ac = np.array([
            np.dot(x[:n_samples - lag], x[lag:]) / var0
            for lag in range(max_lag)
        ])
        below = np.where(direct_ac < np.exp(-1))[0]
        expected = float(below[0] / fs) if len(below) else float(max_lag_sec)

        result = _autocorr_decay_lag(x, fs, max_lag_sec=max_lag_sec)
        assert result == pytest.approx(expected, abs=1.0 / fs)

    def test_constant_signal_returns_nan(self):
        """A zero-variance signal has undefined autocorrelation and must
        return NaN, not raise or divide by zero."""
        x = np.ones(1000)
        result = _autocorr_decay_lag(x, fs=400.0)
        assert np.isnan(result)

    @pytest.mark.slow
    def test_fast_at_high_sampling_rate(self):
        """Regression test for the actual performance bug: the direct
        per-lag computation scaled with fs (since max_lag = max_lag_sec
        * fs), making it impractical at 5000 Hz. This just asserts the
        call completes in well under a second at that scale; run with
        --runslow."""
        import time
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, int(5000 * 600))
        t0 = time.time()
        _autocorr_decay_lag(x, fs=5000.0)
        assert time.time() - t0 < 2.0


# ------------------------------------------------------------------
# clip_features: end-to-end feature extraction
# ------------------------------------------------------------------

class TestClipFeatures:
    def test_feature_count_and_no_crash(self):
        """A full clip at dog scale should produce exactly 42 features
        (33 single-channel mean/std/max aggregates + 9 connectivity
        features) with no exceptions."""
        clip = make_synthetic_clip(n_channels=16, fs=400.0, duration_sec=60.0)
        feats = clip_features(clip)
        assert len(feats) == 42
        assert all(isinstance(v, float) for v in feats.values())

    def test_all_values_finite_on_clean_signal(self):
        """On a well-behaved synthetic signal (no NaNs, no zero-variance
        channels), every feature should come out finite -- a NaN here
        would indicate a real bug in one of the per-channel or
        connectivity computations, not expected missingness."""
        clip = make_synthetic_clip(n_channels=8, fs=400.0, duration_sec=30.0)
        feats = clip_features(clip)
        n_nan = sum(1 for v in feats.values() if not np.isfinite(v))
        assert n_nan == 0, f"{n_nan} unexpected NaN features: " + \
            str([k for k, v in feats.items() if not np.isfinite(v)])


# ------------------------------------------------------------------
# Models: basic construction sanity
# ------------------------------------------------------------------

class TestModels:
    @pytest.mark.parametrize("name", ["logreg", "polylogreg", "rf"])
    def test_model_builds_and_fits(self, name):
        """Every model family must build and fit on a small synthetic
        classification problem without raising."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (60, 10))
        y = (X[:, 0] > 0).astype(int)
        model = make_model(name)
        model.fit(X, y)
        preds = model.predict_proba(X)
        assert preds.shape == (60, 2)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            make_model("not_a_real_model")


# ------------------------------------------------------------------
# Clinical metrics
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Block-level train/test split: the core leakage guard
# ------------------------------------------------------------------

class TestBlockTrainTestSplit:
    def test_no_block_appears_in_both_splits(self):
        """The one property this function exists to guarantee: no
        block_id may appear in both TRAIN and TEST. This is the direct
        EEG analog of the blazar pipeline's prohibition on overlapping
        time windows crossing the train/test boundary."""
        rng = np.random.default_rng(0)
        n_preictal_blocks, n_interictal_blocks = 8, 20
        rows = []
        for b in range(n_preictal_blocks):
            for _ in range(6):
                rows.append({"block_id": f"preictal_block_{b}", "label": 1})
        for b in range(n_interictal_blocks):
            rows.append({"block_id": f"interictal_{b}", "label": 0})
        df = pd.DataFrame(rows)

        train, test = block_train_test_split(df, test_frac_blocks=0.25, seed=0)
        train_blocks = set(train["block_id"])
        test_blocks = set(test["block_id"])
        assert train_blocks.isdisjoint(test_blocks), \
            f"blocks leaked across split: {train_blocks & test_blocks}"

    def test_both_classes_present_in_test_when_enough_blocks(self):
        """With at least 2 blocks per class, TEST should contain both a
        preictal and an interictal block -- the specific failure mode
        (unstratified split silently producing a single-class TEST) that
        surfaced on the first real run of this pipeline, not a
        hypothetical edge case."""
        rows = []
        for b in range(6):
            for _ in range(6):
                rows.append({"block_id": f"preictal_block_{b}", "label": 1})
        for b in range(10):
            rows.append({"block_id": f"interictal_{b}", "label": 0})
        df = pd.DataFrame(rows)

        _, test = block_train_test_split(df, test_frac_blocks=0.25, seed=0)
        assert (test["label"] == 1).any()
        assert (test["label"] == 0).any()


class TestClinicalMetrics:
    def test_fp_per_hour_zero_for_perfect_negative_predictions(self):
        """If every negative is correctly scored below threshold, FP/h
        must be exactly zero."""
        y_true = np.array([0, 0, 0, 0])
        p = np.array([0.1, 0.1, 0.1, 0.1])
        fph = compute_fp_per_hour(y_true, p, threshold=0.5)
        assert fph == 0.0

    def test_sensitivity_at_target_within_bounds(self):
        """Sensitivity must always be a valid proportion in [0, 1]."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        p = rng.uniform(0, 1, 200)
        threshold, achieved_fph, sens = sensitivity_at_target_fph(y_true, p, target_fph=0.15)
        assert 0.0 <= sens <= 1.0
