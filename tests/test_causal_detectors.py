"""Software-only fixtures for causality; none are dataset observations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from awsad.models.statistical import RobustChannelDetector
from awsad.preprocessing.qc_rules import (
    compute_qc_flags,
    dropout_flags,
    flatline_flags,
    spike_flags,
)


def _observations(values):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC"),
        "temperature_c": np.asarray(values, dtype=float),
    })


def _statistical_detector():
    train = _observations(20 + np.random.default_rng(4).normal(size=2000))
    return RobustChannelDetector(stuck_window=8).fit(train, train)


@pytest.mark.parametrize("length", [1, 2, 3, 7, 12, 23, 24, 25, 60, 90])
def test_qc_flags_are_prefix_invariant(length):
    values = 20 + np.random.default_rng(3).normal(size=100)
    values[3:32] = 20
    values[45:50] = 0
    values[62:66] = np.nan
    values[-1] = 10000
    frame = _observations(values)
    expected = compute_qc_flags(frame).iloc[:length]
    actual = compute_qc_flags(frame.iloc[:length])
    pd.testing.assert_frame_equal(actual, expected)


def test_flatline_requires_actual_observed_run_length():
    # The former reverse cumulative sum counted 1+2+...+7 as 28 reports.
    assert not flatline_flags(_observations([20] * 7))["flatline_temperature_c"].any()
    flags = flatline_flags(_observations([20] * 26))["flatline_temperature_c"]
    assert not flags.iloc[:23].any()
    assert flags.iloc[23:].all()


def test_missing_reading_breaks_flatline_evidence():
    flags = flatline_flags(
        _observations([20] * 5 + [np.nan] + [20] * 4), min_len=4,
    )["flatline_temperature_c"]
    assert np.flatnonzero(flags).tolist() == [3, 4, 9]


def test_zero_run_alert_starts_when_evidence_is_available():
    frame = _observations([10, 0, 0, 0, 0, np.nan, 0, 0, 0, 10])
    flags = dropout_flags(frame)["dropout_temperature_c"]
    assert np.flatnonzero(flags).tolist() == [3, 4, 8]


def test_spike_uses_past_scale_and_still_detects_large_change():
    frame = _observations([20, 20.1, 20.2, 20.3, 20.4, 35, 20.5])
    flags = spike_flags(frame)["spike_temperature_c"]
    assert flags.iloc[5]
    with_future = pd.concat([frame, _observations([1e9])], ignore_index=True)
    pd.testing.assert_series_equal(
        flags, spike_flags(with_future)["spike_temperature_c"].iloc[:len(frame)],
    )


@pytest.mark.parametrize("length", [0, 1, 2, 3, 7, 8, 12, 23, 24, 25, 60, 80, 120, 179])
def test_statistical_scores_are_prefix_invariant(length):
    detector = _statistical_detector()
    values = 20 + np.random.default_rng(9).normal(size=180)
    values[:2] = np.nan
    values[20:50] = 20
    values[60:70] = np.nan
    values[120] = 80
    frame = _observations(values)
    full = detector.score(frame, frame)
    prefix = detector.score(frame.iloc[:length], frame.iloc[:length])
    np.testing.assert_allclose(prefix, full[:length], rtol=0, atol=1e-12)
    assert prefix.shape == (length,)
    assert np.isfinite(prefix).all()


def test_statistical_detector_retains_observed_anomaly_evidence():
    detector = _statistical_detector()
    values = 20 + np.random.default_rng(8).normal(size=160)
    values[30] = 80
    values[70:110] = 20
    frame = _observations(values)
    scores = detector.score(frame, frame)
    assert scores[30] > 10
    assert scores[80] >= 3


def test_missing_values_supply_no_fault_evidence_and_stay_missing():
    detector = _statistical_detector()
    frame = _observations([np.nan] * 10 + [20] * 3 + [np.nan] * 20)
    original = frame.copy(deep=True)
    scores = detector.score(frame, frame)
    assert np.all(scores[frame.temperature_c.isna()] == 0)
    # Three readings separated by a gap cannot become an eight-reading plateau.
    assert (scores[10:13] < 3).all()
    pd.testing.assert_frame_equal(frame, original)


def test_fitting_uses_finite_observations_without_mutating_them():
    train = _observations(20 + np.random.default_rng(5).normal(size=300))
    train.loc[20:30, "temperature_c"] = np.nan
    train.loc[100, "temperature_c"] = np.inf
    original = train.copy(deep=True)
    detector = RobustChannelDetector().fit(train, train)
    assert detector.params
    assert all(np.isfinite(v) for p in detector.params.values() for v in p.values())
    pd.testing.assert_frame_equal(train, original)


@pytest.mark.parametrize("architecture", ["lstm", "transformer"])
def test_autoencoder_scores_use_only_completed_trailing_windows(architecture, monkeypatch):
    # A deterministic software double makes future-context contamination
    # observable without training a network or claiming model accuracy.
    if architecture == "lstm":
        from awsad.models.lstm_autoencoder import LSTMAEDetector
        detector = LSTMAEDetector(window=4)
    else:
        from awsad.models.transformer_ae import TransformerAEDetector
        detector = TransformerAEDetector(window=4)

    def window_errors(arr, *_args):
        return np.repeat(arr.sum(axis=(1, 2))[:, None], 4, axis=1)

    monkeypatch.setattr(detector, "_batch_errors", window_errors)
    block = np.arange(1, 31, dtype=np.float32)[:, None]
    full = detector.score_series(block)
    np.testing.assert_array_equal(full[:3], 0)
    assert full[3] == 10
    assert full[4] == 14
    for length in (0, 1, 3, 4, 5, 10, 20, 30):
        np.testing.assert_array_equal(detector.score_series(block[:length]), full[:length])


def test_forecaster_never_copies_future_error_into_warmup(monkeypatch):
    from awsad.models.lstm_forecaster import ForecasterDetector
    detector = ForecasterDetector(window=4)
    detector.err_scale = np.ones(1)
    monkeypatch.setattr(detector, "_predict", lambda batch: batch[:, -1, :])
    block = np.arange(1, 31, dtype=np.float32)[:, None]
    full = detector.score_series(block)
    np.testing.assert_array_equal(full[:4], 0)
    assert full[4] == 1
    for length in (0, 1, 3, 4, 5, 10, 20, 30):
        np.testing.assert_array_equal(detector.score_series(block[:length]), full[:length])


def test_nonfinite_forecast_error_does_not_use_future_score_scale(monkeypatch):
    from awsad.models.lstm_forecaster import ForecasterDetector
    detector = ForecasterDetector(window=4)
    detector.err_scale = np.ones(1)

    def predictions(batch):
        result = batch[:, -1, :].copy()
        result[result == 4] = np.inf
        return result

    monkeypatch.setattr(detector, "_predict", predictions)
    block = np.arange(1, 31, dtype=np.float32)[:, None]
    block[20] = 100
    full = detector.score_series(block)
    assert full[4] == 0
    assert np.isfinite(full).all()
    np.testing.assert_array_equal(detector.score_series(block[:5]), full[:5])
