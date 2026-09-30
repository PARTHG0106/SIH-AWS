"""Synthetic fixtures for software invariants only; never observation evidence."""
import numpy as np
import pandas as pd
import pytest

from awsad.minute_detection import (CHANNELS, SIGNALS, MinuteConfig, apply_thresholds,
    causal_features, compute_signals, fit_thresholds, flatline_durations)


def fixture_frame(n=250):
    t = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC").as_unit("us")
    x = np.arange(n, dtype=float)
    frame = pd.DataFrame({"timestamp": t, "station_id": "fixture", "source": "software_test",
                          "observation_id": [f"fixture-{i}" for i in range(n)]})
    for i, c in enumerate(CHANNELS):
        frame[c] = x * (i + 1) / 10
        frame[c + "__qc_accepted"] = pd.array([True] * n, dtype="boolean")
    return frame


def models():
    return {f"{h}:{c}": None for h in (1, 60) for c in CHANNELS}


def test_future_readings_never_change_past_signals_or_predictions():
    cfg = MinuteConfig()
    frame = fixture_frame()
    before = compute_signals(frame, models(), cfg)
    changed = frame.copy()
    changed.loc[200:, list(CHANNELS)] = 9999
    after = compute_signals(changed, models(), cfg)
    for h in cfg.horizons:
        np.testing.assert_allclose(before[2][h][:200], after[2][h][:200], equal_nan=True)
    for c in CHANNELS:
        for signal in SIGNALS:
            np.testing.assert_allclose(before[0][c][signal][:200], after[0][c][signal][:200], equal_nan=True)


def test_missing_exact_context_disables_forecast_without_filling():
    cfg = MinuteConfig()
    frame = fixture_frame().drop(index=180).reset_index(drop=True)
    f = causal_features(frame, 1, cfg)
    row = frame.index[frame.observation_id.eq("fixture-181")][0]
    assert not f["complete"][row]
    assert pd.isna(f["context_ids"]["1"][row])
    assert len(frame) == 249


def test_gaps_and_missing_readings_reset_flatline_and_abrupt_change():
    frame = fixture_frame(12)
    frame.loc[:, list(CHANNELS)] = 7.
    frame = frame.drop(index=5).reset_index(drop=True)
    frame.loc[8, CHANNELS[0]] = np.nan
    duration, _ = flatline_durations(frame)
    assert list(duration[CHANNELS[1]][:7]) == [0, 1, 2, 3, 4, 0, 1]
    assert np.isnan(duration[CHANNELS[0]][8])
    assert duration[CHANNELS[0]][9] == 0
    signals = compute_signals(frame, models(), MinuteConfig())[0]
    assert np.isnan(signals[CHANNELS[0]]["abrupt_change"][5])


def test_flatline_carries_exact_duration_across_file_boundaries():
    frame = fixture_frame(500)
    frame.loc[:, list(CHANNELS)] = 7.
    full, _ = flatline_durations(frame)
    first, state = flatline_durations(frame.iloc[:250])
    last, _ = flatline_durations(frame.iloc[250:], state)
    for c in CHANNELS:
        np.testing.assert_array_equal(full[c], np.r_[first[c], last[c]])
        assert last[c][-1] == 499


def test_flatline_calibration_rejects_any_unaccepted_context_across_chunks():
    frame = fixture_frame(12)
    frame.loc[:, list(CHANNELS)] = 7.
    frame.loc[3, CHANNELS[0] + "__qc_accepted"] = False
    frame.loc[10:, CHANNELS[0]] = 8.
    _, state, q1 = flatline_durations(frame.iloc[:6], return_quality=True)
    _, _, q2 = flatline_durations(frame.iloc[6:], state, return_quality=True)
    assert q1[CHANNELS[0]].tolist() == [True, True, True, False, False, False]
    assert q2[CHANNELS[0]].tolist() == [False, False, False, False, True, True]
    assert q2[CHANNELS[1]].all()


def test_qc_is_not_a_predictor_and_missing_qc_excludes_fitting():
    frame = fixture_frame()
    before = causal_features(frame, 1, MinuteConfig())
    frame.loc[100, CHANNELS[0] + "__qc_accepted"] = False
    after = causal_features(frame, 1, MinuteConfig())
    np.testing.assert_array_equal(before["X"], after["X"])
    assert before["accepted"][101] and not after["accepted"][101]


def test_sustained_signal_needs_unbroken_real_context():
    frame = fixture_frame(300).drop(index=250).reset_index(drop=True)
    values = compute_signals(frame, models(), MinuteConfig())[0][CHANNELS[0]]["sustained_deviation"]
    pos = frame.index[frame.observation_id.eq("fixture-251")][0]
    assert np.isnan(values[pos:pos + 29]).all()


def test_thresholds_use_strict_exceedance_and_unknown_group_fallback():
    cfg = MinuteConfig(min_calibration_rows=10)
    refs = {"pooled_seen_groups": {c: {s: [np.ones(100)] for s in SIGNALS} for c in CHANNELS}}
    thresholds = fit_thresholds(refs, cfg)
    frame = fixture_frame(3)
    signals = {c: {s: np.array([1., 2., np.nan]) for s in SIGNALS} for c in CHANNELS}
    pred = {h: np.full((3, 3), np.nan) for h in cfg.horizons}
    scored = apply_thresholds(frame, signals, pred, thresholds, cfg)
    assert scored.is_candidate.tolist() == [False, True, False]
    assert scored[CHANNELS[0] + "__alert"].isna().tolist() == [False, False, True]
    assert scored.threshold_group.eq("pooled_seen_groups").all()
    assert scored.hardware_fault_status.eq("unknown").all()
    pd.testing.assert_frame_equal(scored[list(frame)], frame)


@pytest.mark.parametrize("mutate", ["naive", "duplicate", "mixed", "label"])
def test_rejects_ambiguous_inputs(mutate):
    frame = fixture_frame()
    if mutate == "naive":
        frame["timestamp"] = frame.timestamp.dt.tz_localize(None)
    elif mutate == "duplicate":
        frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]
    elif mutate == "mixed":
        frame.loc[1, "station_id"] = "other"
    else:
        frame["label"] = 0
    with pytest.raises(ValueError):
        causal_features(frame, 1, MinuteConfig())
