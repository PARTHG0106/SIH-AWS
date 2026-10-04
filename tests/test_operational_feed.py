"""Generated histories here are software fixtures only, never dataset evidence."""
import numpy as np
import pandas as pd
import pytest
from starlette.applications import Starlette

from app.live_api import make_live_routes
from awsad.demo.operational_feed import SCENARIOS, SOURCE, prepare_operational_feed
from awsad.live_detector import json_safe
from awsad.minute_detection import CHANNELS, SIGNALS
from test_live_api import ASGIClient
from test_live_detector import frame_fixture, make_detector


def baseline_fixture(n=360):
    frame = frame_fixture(n)
    frame["label"] = pd.array([None] * n, dtype="Int8")
    frame["hardware_fault_status"] = "unknown"
    frame["raw_file_sha256"] = "a" * 64
    frame["raw_file_name"] = "software_fixture_only.dat"
    frame["raw_row_number"] = np.arange(n) + 3
    for channel in CHANNELS:
        frame[channel + "__provider"] = "TEST_ONLY"
        frame[channel + "__raw_qc"] = "0"
    return frame


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_each_scenario_preserves_baseline_warmup_missingness_and_unknown_hardware(scenario):
    frame = baseline_fixture()
    frame.loc[190:192, list(CHANNELS)] = np.nan
    frame.loc[185, CHANNELS[0] + "__qc_accepted"] = False
    original = frame.copy(deep=True)
    result = prepare_operational_feed(frame, scenario=scenario)
    pd.testing.assert_frame_equal(frame, original)
    assert result == prepare_operational_feed(frame, scenario=scenario)
    rows = result["observations"]
    assert [row["baseline"] for row in rows] == json_safe(frame.to_dict("records"))
    assert result["group"] == "fixture|synthetic_demo"
    assert result["baseline_group"] == "fixture|software_test"
    assert result["mode"] == "synthetic_demo"
    assert result["threshold_reference"] == "pooled_seen_groups"
    assert result["event"]["modified_rows"] > 0
    assert len({row["observation_id"] for row in rows}) == len(frame)
    assert not set(row["observation_id"] for row in rows).intersection(frame.observation_id)
    assert all(row["label"] is None and row["hardware_fault_status"] == "unknown" for row in rows)
    assert all(row["source"] == SOURCE and row["synthetic"] for row in rows)
    assert not any(row["scenario_modified"] for row in rows[:180])
    for index, row in enumerate(rows):
        assert pd.Timestamp(row["timestamp"]) == frame.timestamp.iloc[index]
        assert row["baseline_raw_file_sha256"] == "a" * 64
        assert row["scenario_modified"] == any(row[c + "__scenario_modified"] for c in CHANNELS)
        assert row["scenario_label"] == (scenario if row["scenario_modified"] else "no_injection")
        for c in CHANNELS:
            assert row[c + "__qc_accepted"] is None
            before = row["baseline_" + c]
            changed = row[c] != before
            assert changed == row[c + "__scenario_modified"]
            if pd.isna(frame[c].iloc[index]):
                assert row[c] is None and not changed
            if index < 180:
                assert row[c] == before


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenarios_adapt_safely_to_less_than_four_hours(scenario):
    result = prepare_operational_feed(baseline_fixture(210), scenario=scenario)
    assert result["rows"] == 210
    event = result["event"]
    assert 1 <= event["parameters"]["duration_minutes"] <= 20
    assert pd.Timestamp(event["scheduled_end_exclusive"]) <= pd.Timestamp(result["end"])
    assert event["modified_rows"] > 0


def test_temperature_humidity_swap_does_not_fill_either_missing_sensor():
    frame = baseline_fixture()
    frame.loc[185, "temperature_c"] = np.nan
    frame.loc[186, "relative_humidity_pct"] = np.nan
    result = prepare_operational_feed(frame, scenario="sensor_swap", channel="pressure_hpa")
    rows = result["observations"]
    assert result["event"]["channels"] == ["temperature_c", "relative_humidity_pct"]
    assert rows[180]["temperature_c"] == frame.relative_humidity_pct.iloc[180]
    assert rows[180]["relative_humidity_pct"] == frame.temperature_c.iloc[180]
    assert rows[185]["temperature_c"] is None and not rows[185]["scenario_modified"]
    assert rows[186]["relative_humidity_pct"] is None and not rows[186]["scenario_modified"]
    assert not any(row["pressure_hpa__scenario_modified"] for row in rows)


@pytest.mark.parametrize("problem", ["three_hours", "gap_in_warmup", "no_target", "invalid_scenario", "invalid_seed", "duplicate", "labels", "known_hardware"])
def test_invalid_or_ineligible_baselines_fail_without_manufacturing_observations(problem):
    frame = baseline_fixture()
    kwargs = {}
    if problem == "three_hours": frame = frame.iloc[:180]
    elif problem == "gap_in_warmup": frame = frame.drop(index=180).reset_index(drop=True)
    elif problem == "no_target": frame["temperature_c"] = np.nan
    elif problem == "invalid_scenario": kwargs["scenario"] = "actual_hardware_failure"
    elif problem == "invalid_seed": kwargs["seed"] = -1
    elif problem == "duplicate": frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif problem == "labels": frame.loc[0, "label"] = 0
    elif problem == "known_hardware": frame.loc[0, "hardware_fault_status"] = "confirmed"
    before = frame.copy(deep=True)
    with pytest.raises(ValueError):
        prepare_operational_feed(frame, **kwargs)
    pd.testing.assert_frame_equal(frame, before)


def test_a_gap_can_delay_intervention_until_a_new_real_warmup_is_complete():
    frame = baseline_fixture().drop(index=175).reset_index(drop=True)
    result = prepare_operational_feed(frame)
    assert result["event"]["scheduled_start"] == "2024-08-01T05:56:00+00:00"
    assert result["rows"] == 359
    assert not any(row["timestamp"] == "2024-08-01T02:55:00+00:00" for row in result["observations"])


def test_live_scorer_is_independent_of_scenario_markers_and_uses_pooled_reference():
    result = prepare_operational_feed(baseline_fixture(), scenario="bias")
    rows = result["observations"]
    stripped = [{key: row[key] for key in ("timestamp", "station_id", "source", "observation_id", *CHANNELS)} for row in rows]
    rich = make_detector(allowed_group=result["group"]).ingest_many(rows)
    bare = make_detector(allowed_group=result["group"]).ingest_many(stripped)
    for original, clean in zip(rich, bare):
        for c in CHANNELS:
            for suffix in (*SIGNALS, "prediction", "prediction_60m", "score", "alert", "reason_codes"):
                assert original[c + "__" + suffix] == clean[c + "__" + suffix]
        assert original["is_candidate"] == clean["is_candidate"]
        assert original["threshold_group"] == "pooled_seen_groups"
        assert original["baseline_group"] == result["baseline_group"]
    assert sum(row["is_candidate"] for row in rich) > 0
    assert sum(row["scenario_modified"] for row in rich) == result["event"]["modified_rows"]


def test_api_prepares_and_scores_synthetic_feed_without_precomputed_scores():
    frame = baseline_fixture()
    frame["anomaly_score"] = 99999.
    frame["is_candidate"] = True
    original = frame.copy(deep=True)
    client = ASGIClient(Starlette(routes=make_live_routes("unused", replay_loader=lambda *args: frame,
                                                        detector_factory=make_detector)))
    query = {"group": "fixture|software_test", "start": "2024-08-01T00:00:00Z",
             "end": "2024-08-01T06:00:00Z", "scenario": "spike", "seed": "12"}
    response = client.get("/api/live/scenario", params=query)
    assert response.status_code == 200, response.json()
    feed = response.json()
    assert "anomaly_score" not in feed["observations"][0]["baseline"]
    session = client.post("/api/live/sessions", json={"group": feed["group"], "expected_cadence_minutes": 1})
    session_id = session.json()["session_id"]
    scored = client.post(f"/api/live/sessions/{session_id}/observations", json={"observations": feed["observations"]})
    assert scored.status_code == 200, scored.json()
    assert len(scored.json()["results"]) == 360
    assert scored.json()["results"][180]["scenario_modified"]
    assert scored.json()["results"][180]["threshold_group"] == "pooled_seen_groups"
    assert scored.json()["results"][180]["hardware_fault_status"] == "unknown"
    short = client.get("/api/live/scenario", params={**query, "end": "2024-08-01T03:00:00Z"})
    assert short.status_code == 400 and "six-hour" in short.json()["error"]
    pd.testing.assert_frame_equal(frame, original)


def test_scenario_endpoint_never_loads_reserved_february_originals():
    def forbidden_loader(*args):
        raise AssertionError("reserved final-source loader was reached")
    client = ASGIClient(Starlette(routes=make_live_routes("unused", replay_loader=forbidden_loader,
                                                        detector_factory=make_detector)))
    response = client.get("/api/live/scenario", params={"group": "bon|noaa_surfrad",
             "start": "2025-02-01T00:00:00Z", "end": "2025-02-01T06:00:00Z"})
    assert response.status_code == 400 and "final-test" in response.json()["error"]
