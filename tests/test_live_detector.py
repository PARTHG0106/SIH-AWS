"""Streaming software invariants; generated fixtures are test-only data."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from awsad.live_detector import LiveMinuteDetector
from awsad.minute_detection import CHANNELS, SIGNALS, MinuteConfig, apply_thresholds, compute_signals


def frame_fixture(n=360, station="fixture"):
    t = np.arange(n, dtype=float)
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-08-01", periods=n, freq="min", tz="UTC"),
                          "station_id": station, "source": "software_test",
                          "observation_id": [f"{station}-{i}" for i in range(n)]})
    for i, c in enumerate(CHANNELS):
        frame[c] = (10., 900., 50.)[i] + np.sin(t / (20 + i * 10))
        frame[c + "__qc_accepted"] = pd.array([True] * n, dtype="boolean")
    return frame


def make_detector(**kwargs):
    cfg = MinuteConfig()
    models = {f"{h}:{c}": None for h in cfg.horizons for c in CHANNELS}
    thresholds = {"pooled_seen_groups": {c: {s: {"threshold": 2. if s != "flatline_minutes" else 90.}
                                                    for s in SIGNALS} for c in CHANNELS}}
    return LiveMinuteDetector(models, thresholds, cfg, **kwargs)


def assert_parity(frame, engine, chunk=1):
    signals, _, predictions, _ = compute_signals(frame, engine.models, engine.config)
    expected = apply_thresholds(frame, signals, predictions, engine.thresholds, engine.config)
    packets = frame.to_dict("records")
    actual = pd.DataFrame([result for start in range(0, len(packets), chunk)
                           for result in engine.ingest_many(packets[start:start + chunk])])
    for c in CHANNELS:
        for suffix in [*SIGNALS, "prediction", "prediction_60m", "score"]:
            col = c + "__" + suffix
            np.testing.assert_allclose(pd.to_numeric(actual[col]).to_numpy(float), expected[col].to_numpy(float),
                                       rtol=1e-12, atol=1e-12, equal_nan=True)
        assert actual[c + "__reason_codes"].tolist() == expected[c + "__reason_codes"].tolist()
        assert actual[c + "__alert"].tolist() == expected[c + "__alert"].astype(object).where(expected[c + "__alert"].notna(), None).tolist()
    assert actual.is_candidate.tolist() == expected.is_candidate.tolist()
    assert actual.reason_codes.tolist() == expected.reason_codes.tolist()
    return actual


@pytest.mark.parametrize("chunk", [1, 31, 360])
def test_stream_batch_parity_including_gaps_missing_and_long_flatline(chunk):
    frame = frame_fixture(620)
    frame.loc[170:500, "temperature_c"] = 7.
    frame.loc[220:225, "pressure_hpa"] = np.nan
    frame.loc[350, [*CHANNELS]] = np.nan
    frame.loc[270, "relative_humidity_pct__qc_accepted"] = False
    frame = frame.drop(index=[299, 300]).reset_index(drop=True)
    engine = make_detector()
    actual = assert_parity(frame, engine, chunk)
    assert actual.loc[frame.timestamp.eq(pd.Timestamp("2024-08-01T08:20Z")), "temperature_c__flatline_minutes"].iloc[0] == 149
    assert engine.snapshot()["groups"]["fixture|software_test"]["buffered_observations"] == 181


def test_interleaved_groups_never_share_history():
    a, b = frame_fixture(200, "a"), frame_fixture(200, "b")
    b.loc[:, list(CHANNELS)] += 20
    engine = make_detector()
    interleaved = []
    for ra, rb in zip(a.to_dict("records"), b.to_dict("records")):
        interleaved.extend([ra, rb])
    out = []
    for start in range(0, len(interleaved), 200):
        out.extend(engine.ingest_many(interleaved[start:start + 200]))
    for station, frame in [("a", a), ("b", b)]:
        separate = make_detector().ingest_many(frame.to_dict("records"))
        actual = [row for row in out if row["station_id"] == station]
        assert actual == separate


def test_bounded_buffers_and_long_flatline_survives_eviction():
    engine = make_detector(health_window_rows=40, max_groups=1)
    frame = frame_fixture(1500)
    frame.loc[:, list(CHANNELS)] = 7.
    for start in range(0, len(frame), 300):
        engine.ingest_many(frame.iloc[start:start + 300].to_dict("records"))
    state = engine._groups["fixture|software_test"]
    assert len(state.history) == len(state.ids) == 181
    assert len(state.health) == 40
    assert all(len(values) == 30 for values in state.residuals.values())
    assert state.latest["temperature_c__flatline_minutes"] == 1499
    with pytest.raises(ValueError, match="capacity"):
        engine.ingest(frame_fixture(1, "other").iloc[0].to_dict())


@pytest.mark.parametrize("mutation", ["duplicate_time", "duplicate_id", "out_of_order", "wrong_group", "inf", "nan_string", "naive", "off_minute", "out_of_range_time", "label"])
def test_bad_packet_is_rejected_without_partial_consumption(mutation):
    engine = make_detector(allowed_group="fixture|software_test")
    rows = frame_fixture(3).to_dict("records")
    engine.ingest(rows[0])
    if mutation == "duplicate_time": rows[2]["timestamp"] = rows[1]["timestamp"]
    elif mutation == "duplicate_id": rows[2]["observation_id"] = rows[0]["observation_id"]
    elif mutation == "out_of_order": rows[2]["timestamp"] = rows[0]["timestamp"]
    elif mutation == "wrong_group": rows[2]["source"] = "other"
    elif mutation == "inf": rows[2]["pressure_hpa"] = float("inf")
    elif mutation == "nan_string": rows[2]["pressure_hpa"] = "NaN"
    elif mutation == "naive": rows[2]["timestamp"] = "2024-08-01 00:02"
    elif mutation == "off_minute": rows[2]["timestamp"] += pd.Timedelta(seconds=1)
    elif mutation == "out_of_range_time": rows[2]["timestamp"] = "3000-01-01T00:00:00Z"
    elif mutation == "label": rows[2]["label"] = 0
    with pytest.raises(ValueError):
        engine.ingest_many(rows[1:])
    assert engine.snapshot()["groups"]["fixture|software_test"]["rows_seen"] == 1


def test_heartbeat_only_infers_missing_slots_with_explicit_cadence():
    row = frame_fixture(1).iloc[0].to_dict()
    engine = make_detector(expected_cadence_minutes=1)
    engine.ingest(row)
    notice = engine.advance("2024-08-01T00:03:00Z")[0]
    assert notice["overdue_slots"] == 3
    assert notice["kind"] == "data_availability"
    assert notice["hardware_fault_status"] == "unknown"
    assert engine.snapshot()["groups"]["fixture|software_test"]["rows_seen"] == 1
    assert engine.advance("2024-08-01T00:03:59Z")[0]["overdue_slots"] == 3
    with pytest.raises(ValueError, match="backwards"):
        engine.advance("2024-08-01T00:02:00Z")
    next_row = {**row, "timestamp": "2024-08-01T00:04:00Z", "observation_id": "later"}
    scored = engine.ingest(next_row)
    assert scored["availability"]["unreported_slots_before"] == 3
    assert scored["temperature_c__abrupt_change"] is None
    assert scored["temperature_c__flatline_minutes"] == 0
    unspecified = make_detector()
    unspecified.ingest(row)
    assert unspecified.advance("2024-08-01T00:03:00Z")[0]["overdue_slots"] is None


def test_missingness_physics_and_confidence_have_separate_semantics():
    engine = make_detector()
    rows = frame_fixture(2).to_dict("records")
    for c in CHANNELS:
        rows[0][c] = None
    missing = engine.ingest(rows[0])
    assert missing["scoring_status"] == "missing_observation"
    assert not missing["is_candidate"]
    assert missing["availability"]["missing_channels"] == list(CHANNELS)
    rows[1]["relative_humidity_pct"] = 120
    result = engine.ingest(rows[1])
    assert result["physics_alerts"][0]["channel"] == "relative_humidity_pct"
    assert result["physics_alerts"][0]["signal"] == "reporting_range"
    assert "supersaturation" in result["physics_alerts"][0]["reason"]
    assert result["severity"] == "high"
    assert result["hardware_fault_status"] == "unknown"
    assert result["type_confidence"] is None


def test_optional_pattern_hook_gets_raw_history_and_failure_does_not_hide_score():
    captured = []
    def hook(history, scored):
        assert "anomaly_score" not in history
        captured.append(len(history))
        raise RuntimeError("test optional model failure")
    engine = make_detector(pattern_predictor=hook)
    result = engine.ingest(frame_fixture(1).iloc[0].to_dict())
    assert result["pattern_evidence"] == {"status": "unavailable", "reason": "RuntimeError"}
    assert result["scoring_status"] == "scored_available_signals"
    assert captured == [1]


def test_frozen_model_parity_on_unchanged_native_observations():
    root = Path(__file__).resolve().parents[1]
    path = root / "data/native_minute_20260928/bon_2024_08.parquet"
    if not path.exists() or not (root / "artifacts_minute_20260928/models.joblib").exists():
        pytest.skip("verified local native observations and frozen model bundle are not installed")
    # No labels or modified observations are generated by this integration test.
    frame = pd.read_parquet(path).iloc[:240].copy()
    engine = LiveMinuteDetector.from_artifacts(root / "artifacts_minute_20260928")
    assert_parity(frame, engine, chunk=120)
