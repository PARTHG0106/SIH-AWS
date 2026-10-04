"""Test-only fixtures for candidate-health denominator and temporal semantics."""
import pandas as pd
import pytest

from awsad.minute_detection import CHANNELS
from awsad.station_health import station_health


def test_unscored_rows_do_not_dilute_candidate_rate():
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC"),
                          "temperature_c": [2., 3., None, None, None],
                          "temperature_c__alert": pd.array([True, False, None, None, None], dtype="boolean")})
    channel = station_health(frame)["channels"]["temperature_c"]
    assert channel["candidate_rate"] == .5
    assert channel["scored_rows"] == 2 and channel["unscored_rows"] == 3
    assert channel["missing_fraction"] == .6
    assert channel["state"] == "critical"


def test_no_scores_means_unknown_not_healthy():
    for frame in [pd.DataFrame(), pd.DataFrame({"temperature_c": [None, None]})]:
        health = station_health(frame)
        assert health["overall_state"] == "unknown"
        assert health["remaining_useful_life"] is None
        assert health["maintenance_advisory"]["state"] == "insufficient_history"
        for channel in health["channels"].values():
            assert channel["candidate_rate"] is None
            assert channel["state"] == channel["trend"] == "unknown"


def test_trend_uses_elapsed_time_not_row_halves_and_is_order_invariant():
    timestamps = ["2024-01-01T00:00Z", "2024-01-01T00:01Z", "2024-01-01T00:02Z", "2024-01-01T01:00Z"]
    frame = pd.DataFrame({"timestamp": pd.to_datetime(timestamps),
                          "temperature_c__alert": [False, False, True, True]})
    health = station_health(frame)
    assert health == station_health(frame.iloc[::-1])
    channel = health["channels"]["temperature_c"]
    assert channel["early_scored_rows"] == 3 and channel["recent_scored_rows"] == 1
    assert channel["early_candidate_rate"] == pytest.approx(1 / 3)
    assert channel["recent_candidate_rate"] == 1
    assert channel["trend"] == "worsening"


def test_availability_counts_received_missing_values_and_absent_reports_separately():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T00:05Z"]),
                          "temperature_c": [None, 3.]})
    assert station_health(frame)["unreported_slots"] is None
    health = station_health(frame, expected_cadence_minutes=1)
    assert health["unreported_slots"] == 4
    assert health["channels"]["temperature_c"]["missing_fraction"] == .5


def test_rising_candidate_activity_produces_inspection_policy_not_failure_forecast():
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=181, freq="min", tz="UTC")})
    for channel in CHANNELS:
        frame[channel] = 10.
        frame[channel + "__alert"] = [False] * 90 + [True] * 91
        frame[channel + "__reason_codes"] = [""] * 90 + ["flatline_minutes"] * 91
    health = station_health(frame)
    advisory = health["maintenance_advisory"]
    assert advisory["state"] == "rising_candidate_activity"
    assert advisory["priority"] == "review_now"
    assert "logger" in advisory["action"]
    assert advisory["history_sufficient_for_policy"]
    assert advisory["evidence"]["scored_rows"] == 181
    assert health["remaining_useful_life"] is None
    assert health["hardware_fault_status"] == "unknown"


def test_missing_feed_has_availability_action_without_maintenance_trend():
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=181, freq="min", tz="UTC"),
                          "temperature_c": [None] * 181})
    advisory = station_health(frame)["maintenance_advisory"]
    assert advisory["priority"] == "check_feed_now"
    assert not advisory["history_sufficient_for_policy"]
