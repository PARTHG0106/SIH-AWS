"""Software-only scenario fixtures; never observational evidence."""
import numpy as np
import pandas as pd
import pytest

from awsad.benchmark.scenarios import CHANNELS, FAULTS, SourceWindow, generate, partition
from awsad.benchmark.operational_model import causal_pattern_features, fit_temperature, temperature_probabilities
from awsad.benchmark.sih_evaluation import evaluate, choose_threshold


def source():
    n = 720
    x = np.arange(n)
    frame = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=n, freq="min", tz="UTC"),
        "temperature_c": 20 + np.sin(x / 30), "pressure_hpa": 990 + np.sin(x / 45),
        "relative_humidity_pct": 60 + 10 * np.sin(x / 60),
        "observation_id": [f"TEST_ONLY_{i}" for i in x], "hardware_fault_status": "unknown"})
    return SourceWindow(frame, "fixture-window", "fit", "software-fixture", "test-only")


@pytest.mark.parametrize("fault", (*FAULTS, "no_injection"))
def test_scenarios_preserve_originals_missingness_and_exact_change_labels(fault):
    original = source()
    original.frame.loc[210:250, "temperature_c"] = np.nan
    before = original.frame.copy(deep=True)
    scenario = generate(original, fault)
    pd.testing.assert_frame_equal(original.frame, before)
    exported = scenario.export()
    pd.testing.assert_frame_equal(exported[before.columns], before)
    raw = before[list(CHANNELS)].to_numpy(float)
    changed = scenario.values[list(CHANNELS)].to_numpy(float)
    assert not np.any(np.isnan(raw) & np.isfinite(changed))
    equal = (raw == changed) | (np.isnan(raw) & np.isnan(changed))
    np.testing.assert_array_equal(scenario.labels != "no_injection", (~equal).any(axis=1))
    assert scenario.event["modified_values"] == (~equal).sum()
    assert exported.hardware_fault_status.eq("unknown").all()


def test_generation_is_deterministic_and_partition_independent_of_values():
    first, second = generate(source(), "drift"), generate(source(), "drift")
    pd.testing.assert_frame_equal(first.values, second.values)
    assert first.event == second.event
    assert partition("bon", "2023-04-01") == "fit"
    assert partition("fpk", "2024-03-01") == "selection"
    assert partition("bon", "2024-09-01") == "calibration"
    assert partition("gwn", "2023-04-01") is None
    assert partition("bon", "2025-01-01") is None
    assert partition("gwn", "2025-02-01") == "test_station"


def test_future_values_cannot_change_current_features():
    frame = source().frame
    before = causal_pattern_features(frame)
    future = frame.copy()
    future.loc[400:, list(CHANNELS)] = -9000
    after = causal_pattern_features(future)
    np.testing.assert_allclose(before.iloc[:400], after.iloc[:400], equal_nan=True)


def test_bounded_live_features_match_batch_and_gap_resets_history():
    frame = source().frame
    whole = causal_pattern_features(frame)
    bounded = causal_pattern_features(frame.iloc[-181:])
    np.testing.assert_allclose(whole.iloc[-1], bounded.iloc[-1], equal_nan=True, atol=1e-6)
    gap = frame.drop(index=400)
    computed = causal_pattern_features(gap)
    assert np.isnan(computed.loc[401, "temperature_c:delta_1"])
    assert np.isnan(computed.loc[401, "temperature_c:past_mean_120"])


def test_long_flatline_correlation_matches_bounded_history():
    frame = source().frame
    frame.loc[200:600, "temperature_c"] = 22.
    whole = causal_pattern_features(frame)
    for end in (330, 413, 442, 610):
        bounded = causal_pattern_features(frame.iloc[end - 180:end + 1])
        np.testing.assert_allclose(whole.iloc[end], bounded.iloc[-1], equal_nan=True, atol=1e-6)


def test_clipping_without_prior_finite_values_does_not_become_dropout():
    original = source()
    trial = generate(original, "clipping")
    channel = trial.event["channels"][0]
    start = int(original.frame.timestamp.searchsorted(pd.Timestamp(trial.event["scheduled_start"])))
    original.frame.loc[start - 120:start - 1, channel] = np.nan
    scenario = generate(original, "clipping")
    assert not scenario.event["applied"]
    assert scenario.event["modified_values"] == 0


def test_calibration_and_evaluation_keep_event_latency_without_point_adjustment():
    scores = np.array([.01, .02, .2, .4, .9, .02])
    labels = np.array(["no_injection"] * 2 + ["drift"] * 3 + ["no_injection"])
    threshold, _ = choose_threshold(labels != "no_injection", scores)
    assert .02 < threshold <= .2
    report = evaluate(labels, scores > .8, scores, None, np.zeros(6, int), np.arange(6),
        [{"fault": "drift", "scenario_id": "TEST_ONLY"}])
    assert report["point"]["recall"] == pytest.approx(1/3)
    assert report["per_fault"]["drift"]["event_recall"] == 1.
    assert report["per_fault"]["drift"]["median_latency_min"] == 2.


def test_temperature_scaling_is_probability_normalized():
    raw = np.array([[.999, .001], [.2, .8], [.99, .01], [.01, .99]])
    value = fit_temperature(raw, [0, 1, 1, 0])
    calibrated = temperature_probabilities(raw, value)
    assert value > 1
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.)
