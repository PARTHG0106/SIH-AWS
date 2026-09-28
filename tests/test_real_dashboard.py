"""Synthetic software fixtures only; never use these records in demonstrations."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.real_dashboard import (CHANNELS, chart_series, load_bundle, missing_intervals,
                                read_observations, review_export, window_summary)


@pytest.fixture
def replay_bundle(tmp_path):
    times = pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z",
                            "2025-01-01T00:04:00Z", "2025-01-01T00:05:00Z"], utc=True)
    frame = pd.DataFrame({
        "timestamp": times, "group": "test|software_test_fixture", "station_id": "test",
        "source": "software_test_fixture", "observation_id": [f"fixture:{i}" for i in range(4)],
        "raw_file_sha256": "a" * 64, "raw_row_number": [3, 4, 7, 8],
        "anomaly_score": [np.nan, 0.5, 1.5, 0.3],
        "is_candidate": pd.array([pd.NA, False, True, False], dtype="boolean"),
        "reason_codes": ["", "", "temperature_c:abrupt_change", ""],
        "scoring_status": ["insufficient_context", "scored", "scored", "scored"],
        "split": "validation", "label": pd.array([pd.NA] * 4, dtype="Int8"),
    })
    for channel in CHANNELS:
        frame[channel] = [12.0, np.nan, 14.0, 15.0]
        frame[f"{channel}__prediction"] = [np.nan, 12.5, np.nan, 14.2]
        frame[f"{channel}__prediction_60m"] = np.nan
        frame[f"{channel}__alert"] = pd.array([pd.NA, False, True, False], dtype="boolean")
        frame[f"{channel}__reason_codes"] = ["", "", "abrupt_change", ""]
        frame[f"{channel}__raw_qc"] = ["0", "9", "1", "0"]
        frame[f"{channel}__qc_code"] = [0, 9, 1, 0]
        frame[f"{channel}__raw_value"] = ["12.0", "-9999.9", "14.0", "15.0"]
    for name in ("metrics", "detector"):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "dataset_policy": "real_observations_only", "fixture": True,
            "hardware_fault_metrics": "unavailable_without_reviewed_evidence",
        }))
    (tmp_path / "scored_observations").mkdir()
    frame.to_parquet(tmp_path / "scored_observations" / "test_202501.parquet", index=False)
    pd.DataFrame([{"event_id": "test-event", "group": "test|software_test_fixture",
                   "station_id": "test", "source": "software_test_fixture",
                   "start": times[2], "end": times[2], "channel": "temperature_c",
                   "max_score": 1.5, "reason_codes": "abrupt_change",
                   "review_status": "unknown"}]).to_csv(tmp_path / "candidate_events.csv", index=False)
    return tmp_path, frame


def test_replay_rejects_legacy_artifacts(tmp_path):
    (tmp_path / "metrics.json").write_text('{"test": {"f1": 0.99}}')
    with pytest.raises(ValueError, match="Legacy injected-data"):
        load_bundle(tmp_path)


def test_replay_reads_only_requested_group_and_time_without_repair(replay_bundle):
    path, original = replay_bundle
    bundle = load_bundle(path)
    assert bundle["catalog"].iloc[0]["records"] == 4
    actual = read_observations(bundle, "test|software_test_fixture",
                               "2025-01-01T00:01:00Z", "2025-01-01T00:04:00Z")
    assert actual["observation_id"].tolist() == ["fixture:1", "fixture:2"]
    assert pd.isna(actual.iloc[0]["temperature_c"])
    assert actual["temperature_c__raw_value"].tolist() == ["-9999.9", "14.0"]
    assert actual["temperature_c__qc_code"].tolist() == [9, 1]
    assert actual["label"].isna().all()
    assert read_observations(bundle, "other|software_test_fixture",
                             original.timestamp.min(), original.timestamp.max()).empty


def test_chart_breaks_at_absent_rows_and_values_and_separates_predictions(replay_bundle):
    _, frame = replay_bundle
    chart = chart_series(frame, "temperature_c")
    observed = chart.loc[chart.series.eq("Observed")]
    predicted = chart.loc[chart.series.str.startswith("Model:")]
    assert observed["value"].tolist() == [12.0, 14.0, 15.0]
    assert observed["segment"].iloc[0] != observed["segment"].iloc[1]
    assert observed["segment"].iloc[1] == observed["segment"].iloc[2]
    assert predicted["value"].tolist() == [12.5, 14.2]
    assert predicted["segment"].nunique() == 2
    assert set(chart.timestamp).issubset(set(frame.timestamp))
    pd.testing.assert_series_equal(frame["temperature_c"], pd.Series(
        [12.0, np.nan, 14.0, 15.0], name="temperature_c"))


def test_gap_and_candidate_counts_keep_unknown_separate(replay_bundle):
    _, frame = replay_bundle
    assert missing_intervals(frame)["unreported_minute_slots"].tolist() == [2]
    assert window_summary(frame) == {"records": 4, "candidates": 1, "scored": 3,
                                     "missing_values": 3, "unreported_slots": 2}


def test_review_export_never_promotes_candidates_to_truth(replay_bundle):
    path, _ = replay_bundle
    review = review_export(load_bundle(path), "test|software_test_fixture")
    assert review["review_status"].tolist() == ["unknown"]
    assert review["evidence_url_or_path"].tolist() == [""]


def test_real_dashboard_app_renders_candidates_and_empty_windows(replay_bundle, monkeypatch):
    from streamlit.testing.v1 import AppTest

    path, source = replay_bundle
    # An extra isolated timestamp makes a genuine empty day selectable.
    extra = source.tail(1).copy()
    extra["timestamp"] = pd.Timestamp("2025-01-03T00:00:00Z")
    extra["observation_id"] = "fixture:later"
    pd.concat([source, extra], ignore_index=True).to_parquet(
        path / "scored_observations" / "test_202501.parquet", index=False)
    monkeypatch.setenv("SKYGUARD_ARTIFACTS", str(path))
    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Observation replay"
    assert [metric.value for metric in app.metric] == ["4", "1", "3", "2"]
    assert len(app.tabs) == 3
    assert len(app.get("vega_lite_chart")) == 3
    assert any("Hardware-fault status remains unknown" in message.value for message in app.info)
    app.sidebar.radio[0].set_value("Calendar window").run()
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 3
    app.sidebar.date_input[0].set_value(pd.Timestamp("2025-01-02").date()).run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["0", "0", "0", "0"]
    assert not app.get("vega_lite_chart")
    assert any("No archived records" in message.value for message in app.warning)


def test_real_dashboard_reports_missing_artifacts_without_simulation(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SKYGUARD_ARTIFACTS", str(tmp_path / "missing"))
    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.error) == 1
    assert "does not exist" in app.error[0].value
    assert not app.get("vega_lite_chart")
