"""Synthetic software fixtures only; never package these as station observations."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import ThreadpoolController

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awsad.real_observation_pipeline import (CHANNELS, RealObservationConfig,
    _predict, _provider_qc_agreement, build_causal_features, run_real_observation_pipeline)


def fixture_observations(hours=400, stations=("A", "B", "C")):
    time = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rows = []
    x = np.arange(hours)
    for station_index, station in enumerate(stations):
        frame = pd.DataFrame({
            "observation_id": [f"{station}-{i}" for i in x], "timestamp": time,
            "station_id": station, "source": "software_test_fixture",
            "temperature_c": 10 + station_index + 5 * np.sin(2 * np.pi * x / 24) + x * .002,
            # Deliberately high-altitude-like values: no sea-level pressure range filter.
            "pressure_hpa": 450 + 5 * station_index + np.sin(x / 18),
            "relative_humidity_pct": 55 + 10 * np.cos(2 * np.pi * x / 24),
        })
        for channel in CHANNELS:
            frame[channel + "__qc_accepted"] = pd.Series(True, index=frame.index, dtype="boolean")
            frame[channel + "__qc_rejected"] = pd.Series(False, index=frame.index, dtype="boolean")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def small_config(**kwargs):
    cfg = RealObservationConfig(thinning_minutes=None, holdout_station_ids=("C",),
        max_train_rows=80, max_validation_rows=60, min_train_rows=20,
        min_validation_rows=5, max_iter=4, min_samples_leaf=8, max_cpu_threads=1)
    return replace(cfg, **kwargs)


@pytest.mark.parametrize("horizon", [60, 1440])
def test_exact_time_features_are_prefix_invariant_and_exclude_qc(horizon):
    frame = fixture_observations(160, stations=("A",))
    full = build_causal_features(frame, horizon)
    prefix = build_causal_features(frame.iloc[:100].copy(), horizon)
    np.testing.assert_allclose(full["X"][:100], prefix["X"], equal_nan=True)
    changed = frame.copy()
    changed.loc[100:, list(CHANNELS)] = 1e6
    future_changed = build_causal_features(changed, horizon)
    np.testing.assert_allclose(full["X"][:100], future_changed["X"][:100], equal_nan=True)
    assert not any("qc" in name or "label" in name for name in full["feature_names"])
    assert np.all(full["origin"] < frame.timestamp)
    assert full["complete"][80]
    pressure_feature = full["feature_names"].index("pressure_hpa__origin_minus_0m")
    assert full["X"][80, pressure_feature] < 500  # Actual station pressure retained.
    lookup = frame.set_index("observation_id").timestamp
    for lineage in full["lineage"].values():
        for position in (80, 100):
            assert lookup.loc[lineage[position]] <= full["origin"].iloc[position]


def test_missing_native_report_does_not_become_a_positional_lag():
    frame = fixture_observations(100, stations=("A",))
    gap_time = frame.timestamp.iloc[70]
    frame = frame.loc[frame.timestamp != gap_time].reset_index(drop=True)
    features = build_causal_features(frame, 60)
    for hour_after_gap in (1, 2):
        pos = np.flatnonzero((frame.timestamp == gap_time + pd.Timedelta(hours=hour_after_gap)).to_numpy())[0]
        assert not features["complete"][pos]
    assert features["complete"][60]


def test_selection_run_does_not_inspect_test_targets_or_fit_holdout(tmp_path):
    observations = fixture_observations()
    cfg = small_config(evaluate_test=False)
    first = run_real_observation_pipeline(observations, tmp_path / "first", cfg)
    changed = observations.copy()
    test_start = pd.Timestamp(first["coverage"]["validation_end_exclusive"])
    test_mask = changed.timestamp >= test_start
    # These invalid values would raise if test targets or QC were inspected.
    for channel in CHANNELS:
        changed[channel] = changed[channel].astype(object)
        changed.loc[test_mask, channel] = "do not read a test target during selection"
        changed[channel + "__qc_accepted"] = changed[channel + "__qc_accepted"].astype(object)
        changed.loc[test_mask, channel + "__qc_accepted"] = "do not inspect test quality"
    second = run_real_observation_pipeline(changed, tmp_path / "second", cfg)
    assert first["data_fingerprint"] == second["data_fingerprint"]
    assert first["horizons"] == second["horizons"]
    assert not first["coverage"]["test_values_inspected"]
    scored = pd.read_parquet(tmp_path / "first/scored_observations.parquet")
    assert "test" not in set(scored.evaluation_split)
    trace = json.loads((tmp_path / "first/training_trace.json").read_text())
    times = observations.set_index("observation_id").timestamp
    train_end = pd.Timestamp(first["coverage"]["train_end_exclusive"])
    for channel in CHANNELS:
        channel_trace = trace["60m"][channel]
        for partition, identifiers in channel_trace.items():
            assert identifiers and all(not identifier.startswith("C-") for identifier in identifiers)
            if partition in ("fit", "training_scale"):
                assert (times.loc[identifiers] < train_end).all()
            else:
                assert (times.loc[identifiers] >= train_end).all()
                assert (times.loc[identifiers] < test_start).all()
        assert set(channel_trace["fit"]).isdisjoint(channel_trace["training_scale"])
        assert set(channel_trace["validation_selection"]).isdisjoint(channel_trace["validation_calibration"])
        assert times.loc[channel_trace["fit"]].max() < times.loc[channel_trace["training_scale"]].min()
        assert times.loc[channel_trace["validation_selection"]].max() < times.loc[channel_trace["validation_calibration"]].min()


def test_scored_records_preserve_missing_values_unknown_labels_and_rejected_targets(tmp_path):
    observations = fixture_observations()
    missing_row = observations.index[observations.observation_id == "A-365"][0]
    unknown_row = observations.index[observations.observation_id == "B-367"][0]
    rejected_row = observations.index[observations.observation_id == "C-368"][0]
    observations.loc[missing_row, "relative_humidity_pct"] = np.nan
    observations.loc[unknown_row, "temperature_c__qc_accepted"] = pd.NA
    observations.loc[unknown_row, "temperature_c__qc_rejected"] = pd.NA
    observations.loc[rejected_row, "pressure_hpa__qc_accepted"] = False
    observations.loc[rejected_row, "pressure_hpa__qc_rejected"] = True
    observations["fault_label"] = pd.Series(pd.NA, index=observations.index, dtype="string")
    before = observations.copy(deep=True)
    report = run_real_observation_pipeline(observations, tmp_path / "final", small_config())
    pd.testing.assert_frame_equal(observations, before)
    scored = pd.read_parquet(tmp_path / "final/scored_observations.parquet").set_index("observation_id")
    trace = json.loads((tmp_path / "final/training_trace.json").read_text())["60m"]
    assert pd.isna(scored.loc["A-365", "relative_humidity_pct"])
    assert pd.isna(scored.loc["A-365", "60m__relative_humidity_pct__flag"])
    assert pd.isna(scored.loc["A-365", "60m__flag"])
    assert pd.isna(scored.loc["B-367", "temperature_c__qc_accepted"])
    assert scored["fault_label"].isna().all() and "label" not in scored
    assert np.isfinite(scored.loc["C-368", "60m__pressure_hpa__anomaly_score"])
    for channel in CHANNELS:
        results = report["horizons"]["60m"]["channels"][channel]
        assert results["selected_model"] == min(results["validation_candidates"], key=lambda x: results["validation_candidates"][x]["macro_group_mae"])
        assert results["evaluation"]["test"]["seen"]["provider_qc_agreement"]["status"] == "unavailable"
        assert "persistence" in results["evaluation"]["test"]["holdout"]["provider_accepted_forecasting"]
        assert set(results["per_group_residual_scales"]) == {"A|software_test_fixture", "B|software_test_fixture"}
        assert set(results["per_group_thresholds"]) == {"A|software_test_fixture", "B|software_test_fixture"}
        for station in ("A", "B"):
            group = station + "|software_test_fixture"
            scale_ids = [identifier for identifier in trace[channel]["training_scale"] if identifier.startswith(station + "-")]
            calibration_ids = [identifier for identifier in trace[channel]["validation_calibration"] if identifier.startswith(station + "-")]
            assert (scored.loc[scale_ids, "timestamp"] < pd.Timestamp(report["coverage"]["train_end_exclusive"])).all()
            assert (scored.loc[calibration_ids, "timestamp"] >= pd.Timestamp(report["coverage"]["train_end_exclusive"])).all()
            assert (scored.loc[calibration_ids, "timestamp"] < pd.Timestamp(report["coverage"]["validation_end_exclusive"])).all()
            expected_scale = max(float(np.median(scored.loc[scale_ids, f"60m__{channel}__absolute_error"])), small_config().scale_epsilon)
            expected_threshold = float(np.quantile(scored.loc[calibration_ids, f"60m__{channel}__anomaly_score"], small_config().anomaly_quantile))
            assert results["per_group_residual_scales"][group] == pytest.approx(expected_scale)
            assert results["per_group_thresholds"][group] == pytest.approx(expected_threshold)
        # Held-out station gets pooled parameters, never its own learned map.
        assert scored.loc["C-368", f"60m__{channel}__residual_scale"] == results["residual_scale"]
        assert scored.loc["C-368", f"60m__{channel}__threshold"] == results["threshold"]
    aggregate = report["horizons"]["60m"]["aggregate"]
    assert set(aggregate["per_group_thresholds"]) == {"A|software_test_fixture", "B|software_test_fixture"}
    for station in ("A", "B"):
        identifiers = [identifier for identifier in trace["aggregate_calibration"] if identifier.startswith(station + "-")]
        assert (scored.loc[identifiers, "timestamp"] >= pd.Timestamp(report["coverage"]["train_end_exclusive"])).all()
        assert (scored.loc[identifiers, "timestamp"] < pd.Timestamp(report["coverage"]["validation_end_exclusive"])).all()
        expected = np.quantile(scored.loc[identifiers, "60m__anomaly_score"], small_config().anomaly_quantile)
        assert aggregate["per_group_thresholds"][station + "|software_test_fixture"] == pytest.approx(expected)
    assert scored.loc["C-368", "60m__threshold"] == aggregate["threshold"]
    serialized = json.dumps(report)
    assert '"f1"' not in serialized and '"accuracy"' not in serialized
    assert '"false_positive_rate"' not in serialized
    assert report["horizons"]["60m"]["aggregate"]["evaluation"]["test"]["seen"]["unavailable_score_rows"] > 0


def test_provider_qc_agreement_keeps_unknowns_out_of_comparison():
    frame = pd.DataFrame({"temperature_c__qc_accepted": pd.array([True, False, pd.NA], dtype="boolean"),
                          "temperature_c__qc_rejected": pd.array([False, True, pd.NA], dtype="boolean")})
    flags = pd.Series([False, True, True], dtype="boolean")
    definitions = {"qc_agreement": {"documented": True, "positive_definition": "software-test rejected flag",
                    "negative_definition": "software-test accepted flag", "evidence": "test fixture only"}}
    report = _provider_qc_agreement(frame, "temperature_c", flags, definitions)
    assert report["known_scored_rows"] == 2
    assert report["unknown_or_unscored_rows"] == 1
    assert report["provider_rejected_flagged"] == 1
    assert report["provider_accepted_unflagged"] == 1
    assert report["task"] == "provider_QC_agreement_not_hardware_fault_truth"


def test_naive_timestamps_are_not_assumed_utc(tmp_path):
    observations = fixture_observations()
    observations["timestamp"] = observations.timestamp.dt.tz_localize(None)
    with pytest.raises(ValueError, match="explicit timezone"):
        run_real_observation_pipeline(observations, tmp_path / "invalid", small_config())


def test_cli_refuses_unverified_input_before_parquet_or_ml_imports(tmp_path):
    dataset = tmp_path / "legacy_dataset"
    dataset.mkdir()
    (dataset / "source_metadata.json").write_text(json.dumps({"status": "legacy_local_snapshot"}))
    runner = Path(__file__).resolve().parents[1] / "scripts/run_real_observation_train.py"
    script = """import builtins, runpy, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'pandas', 'sklearn', 'torch'} or name == 'awsad.real_observation_pipeline':
        raise AssertionError('unverified input reached data/model imports: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
"""
    script += "sys.argv = " + repr([str(runner), "--dataset-dir", str(dataset), "--out", str(tmp_path / "guard_output")]) + "\n"
    script += "try:\n    runpy.run_path(" + repr(str(runner)) + ", run_name='__main__')\n"
    script += "except SystemExit as exc:\n    assert exc.code == 2, exc.code\nelse:\n    raise AssertionError('unverified data accepted')\n"
    check = tmp_path / "check_cli_guard.py"
    check.write_text(script, encoding="utf-8")
    result = subprocess.run([sys.executable, str(check)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    preflight = json.loads((tmp_path / "guard_output/training_preflight.json").read_text())
    assert preflight["eligible"] is False
    assert not (tmp_path / "guard_output/forecast_models.joblib").exists()


def test_residual_prediction_adds_change_to_original_origin_and_applies_fixed_shrinkage():
    class ChangePredictor:
        def predict(self, values):
            return np.full(len(values), 2.0)
    bundle = {"hist_gradient_boosting_residual": ChangePredictor(), "residual_shrinkage": .5,
              "max_cpu_threads": 1, "threadpool_controller": ThreadpoolController()}
    features = {"X": np.ones((2, 1)), "persistence": np.array([[450.0], [451.0]])}
    np.testing.assert_array_equal(_predict("hist_gradient_boosting_residual", bundle, features, 0, np.array([0, 1])),
                                  [452, 453])
    np.testing.assert_array_equal(_predict("hist_gradient_boosting_residual_shrunk", bundle, features, 0, np.array([0, 1])),
                                  [451, 452])
    np.testing.assert_array_equal(features["persistence"].ravel(), [450, 451])


def test_residual_training_targets_are_differences_of_the_traced_observations(tmp_path, monkeypatch):
    import awsad.real_observation_pipeline as pipeline
    captured = []
    class RecordingEstimator:
        def __init__(self, **parameters):
            self.parameters = parameters
        def fit(self, values, targets):
            captured.append(np.asarray(targets).copy())
            self.center = float(np.median(targets))
            return self
        def predict(self, values):
            return np.full(len(values), self.center)
        def get_params(self):
            return self.parameters
    monkeypatch.setattr(pipeline, "HistGradientBoostingRegressor", RecordingEstimator)
    # This test records fit targets, not model serialization of the test double.
    monkeypatch.setattr(pipeline.joblib, "dump", lambda *args, **kwargs: None)
    observations = fixture_observations(200)
    pipeline.run_real_observation_pipeline(observations, tmp_path / "trace", small_config(evaluate_test=False))
    trace = json.loads((tmp_path / "trace/training_trace.json").read_text())["60m"]
    by_id = observations.set_index("observation_id")
    by_time = observations.set_index(["source", "station_id", "timestamp"])
    assert len(captured) == 2 * len(CHANNELS)
    for channel_index, channel in enumerate(CHANNELS):
        target_records = by_id.loc[trace[channel]["fit"]]
        target_values = target_records[channel].to_numpy()
        origin_values = np.array([by_time.loc[(row.source, row.station_id,
                                   row.timestamp - pd.Timedelta(hours=1)), channel]
                                  for row in target_records.itertuples()])
        np.testing.assert_allclose(captured[2 * channel_index], target_values)
        np.testing.assert_allclose(captured[2 * channel_index + 1], target_values - origin_values)
