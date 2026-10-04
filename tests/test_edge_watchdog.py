"""Synthetic software fixtures only; no fixture is an observation or fault label."""
import json

import pytest

from edge.watchdog import CHANNELS, MAX_COUNTER, PreliminaryWatchdog


def packet(temp=20.0, pressure=1000.0, rh=50.0):
    return dict(zip(CHANNELS, (temp, pressure, rh)))


def test_missing_is_unavailable_and_resets_temporal_context():
    watch = PreliminaryWatchdog("fixture|synthetic_test", 60)
    first = watch.ingest(0, packet())
    assert first["channels"]["temperature_c"]["step_alert"] is None
    missing = watch.ingest(60, packet(temp=None))
    channel = missing["channels"]["temperature_c"]
    assert channel["observed"] is None and channel["availability"] == "missing"
    assert channel["physical_domain_alert"] is None and channel["step_alert"] is None
    assert missing["hardware_fault_status"] == "unknown"
    returned = watch.ingest(120, packet(temp=35))
    assert returned["channels"]["temperature_c"]["step_alert"] is None


def test_physical_step_and_flatline_are_separate_review_signals():
    watch = PreliminaryWatchdog("fixture|synthetic_test", 60, flatline_seconds=120)
    watch.ingest(0, packet())
    step = watch.ingest(60, packet(temp=27, rh=120))
    assert step["channels"]["temperature_c"]["step_alert"] is True
    assert step["channels"]["relative_humidity_pct"]["physical_domain_alert"] is False
    assert step["channels"]["relative_humidity_pct"]["reporting_range_alert"] is True
    assert any(alert["signal"] == "reporting_range" and "supersaturation" in alert["reason"] for alert in step["alerts"])
    result = watch.ingest(120, packet(temp=27))
    assert result["channels"]["pressure_hpa"]["flatline_alert"] is False
    assert result["channels"]["relative_humidity_pct"]["step_alert"] is None
    assert result["hardware_fault_status"] == "unknown"
    assert watch.ingest(180, packet(temp=27))["channels"]["pressure_hpa"]["flatline_alert"] is True


def test_gap_breaks_step_and_flatline_and_heartbeat_does_not_fill():
    watch = PreliminaryWatchdog("fixture|synthetic_test", 60, flatline_seconds=120, grace_seconds=5)
    assert watch.heartbeat(0)["unreported_slots"] is None
    watch.ingest(0, packet())
    assert watch.heartbeat(64)["unreported_slots"] == 0
    assert watch.heartbeat(65)["unreported_slots"] == 1
    assert watch.heartbeat(65)["unreported_slots"] == 1
    assert watch.rows_seen == 1
    result = watch.ingest(180, packet(temp=45))
    assert result["availability"]["unreported_slots_before"] == 2
    assert result["channels"]["temperature_c"]["step_alert"] is None
    assert result["channels"]["pressure_hpa"]["flatline_duration_seconds"] == 0
    assert watch.heartbeat(180)["state"] == "no_packet_overdue"


def test_repetition_uses_anchor_not_chaining_small_changes():
    watch = PreliminaryWatchdog("fixture|synthetic_test", 60, flatline_seconds=120, flatline_epsilon=.1)
    watch.ingest(0, packet(temp=20))
    watch.ingest(60, packet(temp=20.08))
    result = watch.ingest(120, packet(temp=20.16))
    assert result["channels"]["temperature_c"]["flatline_duration_seconds"] == 0
    assert result["channels"]["temperature_c"]["flatline_alert"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "20"])
def test_invalid_packet_is_atomic(value):
    watch = PreliminaryWatchdog("fixture|synthetic_test", 60)
    watch.ingest(0, packet())
    previous = watch.snapshot()
    with pytest.raises(ValueError):
        watch.ingest(60, packet(rh=value))
    assert watch.snapshot() == previous


def test_order_group_and_heartbeat_validation():
    watch = PreliminaryWatchdog("fixture|synthetic_test", 60)
    watch.ingest(60, packet())
    for time in (60, 59):
        with pytest.raises(ValueError):
            watch.ingest(time, packet())
    with pytest.raises(ValueError):
        watch.ingest(120, packet(), station_source="other|synthetic_test")
    with pytest.raises(ValueError):
        watch.heartbeat(59)
    watch.heartbeat(180)
    # Delayed but ordered originals remain admissible after a receiver heartbeat.
    assert watch.ingest(120, packet())["rows_seen"] == 2
    with pytest.raises(ValueError):
        watch.heartbeat(179)


def test_state_is_constant_size_and_counter_saturates():
    watch = PreliminaryWatchdog("fixture|synthetic_test", 60)
    keys = set(watch.snapshot())
    for i in range(5000):
        watch.ingest(i * 60, packet(temp=20 + i % 7))
    state = watch.snapshot()
    assert set(state) == keys
    assert sum(len(state[key]) for key in ("last_values", "flatline_anchors", "flatline_starts")) == 9
    assert state["state_bounds"]["historical_packet_buffer"] == 0
    json.dumps(state, allow_nan=False)
    state["last_values"][0] = 900
    assert watch.last_values[0] != 900
    watch.rows_seen = MAX_COUNTER
    watch.ingest(5000 * 60, packet())
    assert watch.rows_seen == MAX_COUNTER


@pytest.mark.parametrize("kwargs", [{"cadence_seconds": 0}, {"cadence_seconds": 60, "grace_seconds": 60},
                                  {"cadence_seconds": 60, "step_limits": (1, 2)},
                                  {"cadence_seconds": 60, "flatline_epsilon": -1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PreliminaryWatchdog("fixture|synthetic_test", **kwargs)


def test_jsonl_adapter_keeps_original_evidence_and_separates_heartbeat(tmp_path):
    from types import SimpleNamespace
    from edge.run_watchdog import run_jsonl

    observation = {**packet(temp=None), "timestamp": "2025-01-01T00:00:00Z",
                   "station_id": "fixture", "source": "synthetic_test",
                   "temperature_c__raw_qc": "fixture_only", "observation_id": "fixture-original-1"}
    heartbeat = {"kind": "heartbeat", "timestamp": "2025-01-01T00:02:00Z"}
    source, output = tmp_path / "fixture.jsonl", tmp_path / "results.jsonl"
    source.write_text(json.dumps(observation) + "\n" + json.dumps(heartbeat) + "\n")
    run_jsonl(SimpleNamespace(group="fixture|synthetic_test", cadence=60, grace=0,
                             input=str(source), output=str(output)))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["original"] == observation
    assert rows[0]["result"]["channels"]["temperature_c"]["availability"] == "missing"
    assert rows[1]["kind"] == "availability"
    assert rows[1]["result"]["unreported_slots"] == 2
    assert "original" not in rows[1]


def test_iso_adapter_requires_timezone():
    from edge.run_watchdog import _stamp
    with pytest.raises(ValueError, match="timezone"):
        _stamp({"timestamp": "2025-01-01T00:00:00"})
    assert _stamp({"timestamp": "2025-01-01T05:30:00+05:30"}) == _stamp({"timestamp": "2025-01-01T00:00:00Z"})


def test_per_channel_flatline_thresholds_use_strict_elapsed_comparison():
    watch = PreliminaryWatchdog("fixture|synthetic_test", 60, flatline_seconds=(60, 120, 180))
    watch.ingest(0, packet())
    equal = watch.ingest(60, packet())
    assert all(equal["channels"][c]["flatline_alert"] is False for c in CHANNELS)
    second = watch.ingest(120, packet())
    assert [second["channels"][c]["flatline_alert"] for c in CHANNELS] == [True, False, False]
    third = watch.ingest(180, packet())
    assert [third["channels"][c]["flatline_alert"] for c in CHANNELS] == [True, True, False]
    assert watch.snapshot()["policy_origin"] == "provided_operator_policy"
    assert watch.snapshot()["comparison"] == "strictly_greater"


def test_step_threshold_is_strict_and_zero_reference_is_supported():
    watch = PreliminaryWatchdog("fixture|synthetic_test", 60, step_limits=(5, 6, 20))
    watch.ingest(0, packet())
    assert watch.ingest(60, packet(temp=25))["channels"]["temperature_c"]["step_alert"] is False
    assert watch.ingest(120, packet(temp=30.1))["channels"]["temperature_c"]["step_alert"] is True
    zero = PreliminaryWatchdog("fixture|synthetic_test", 60, flatline_seconds=0)
    zero.ingest(0, packet())
    assert zero.ingest(60, packet())["channels"]["pressure_hpa"]["flatline_alert"] is True
    assert PreliminaryWatchdog("fixture|synthetic_test", 60).snapshot()["policy_origin"] == "default_operator_policy"


def _frozen_fixture(tmp_path):
    """Constructed policy fixture only; never a real calibration artifact."""
    source = {channel: {"abrupt_change": {"threshold": index + .5, "comparison": "strictly_greater", "reference_rows": 100},
                        "flatline_minutes": {"threshold": (index + 1) * 10, "comparison": "strictly_greater", "reference_rows": 100}}
              for index, channel in enumerate(CHANNELS)}
    data = {"frozen_at_utc": "2024-01-01T00:00:00Z", "test_used_for_selection": False,
            "thresholds": {"fixture|synthetic_test": source, "pooled_seen_groups": source},
            "dataset_policy": "synthetic_unit_test_only", "config": {"calibration_end": "2023-12-01T00:00:00Z"}}
    path = tmp_path / "fixture_detector.json"
    path.write_text(json.dumps(data))
    return path


def test_frozen_policy_exports_per_channel_units_and_declares_pooled_reference(tmp_path):
    import hashlib
    from edge.export_policy import export_policy
    from edge.run_watchdog import _new_watch
    from types import SimpleNamespace

    path = _frozen_fixture(tmp_path)
    policy = export_policy(path, ["fixture|synthetic_test", "unseen|synthetic_test"])
    assert policy["source"]["detector_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert policy["groups"]["fixture|synthetic_test"]["configuration"]["flatline_seconds"] == [600, 1200, 1800]
    assert policy["groups"]["unseen|synthetic_test"]["threshold_group"] == "pooled_seen_groups"
    args = SimpleNamespace(cadence=60, grace=0)
    watch = _new_watch(args, "fixture|synthetic_test", policy)
    assert watch.step_limits == (.5, 1.5, 2.5)
    assert watch.snapshot()["policy_origin"] == "frozen_signal_threshold_export"
    with pytest.raises(ValueError, match="absent"):
        _new_watch(args, "other|synthetic_test", policy)
    with pytest.raises(ValueError, match="cadence"):
        _new_watch(SimpleNamespace(cadence=30, grace=0), "fixture|synthetic_test", policy)


def test_frozen_export_rejects_incompatible_threshold_comparison(tmp_path):
    from edge.export_policy import export_policy
    path = _frozen_fixture(tmp_path)
    data = json.loads(path.read_text())
    data["thresholds"]["fixture|synthetic_test"]["temperature_c"]["abrupt_change"]["comparison"] = "greater_or_equal"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="strictly_greater"):
        export_policy(path)
