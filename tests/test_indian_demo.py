"""Synthetic software fixtures for the isolated demo adapter, never observations."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from awsad.demo.indian_stations import (
    CHANNELS, DATASET_POLICY, build_demo_bundle, load_demo_bundle,
    parse_native_station, read_demo_station, simulate_scenario,
)


SID = "42182099999"
TEMPERATURE = "baseline_temperature_c"
PRESSURE = "baseline_sea_level_pressure_hpa"
RH = "derived_relative_humidity_pct"


@pytest.fixture
def native_files(tmp_path):
    """Deliberately irregular, out-of-order duplicate reports with missing/QC data."""
    raw_dir = tmp_path / "originals"
    (raw_dir / "2024").mkdir(parents=True)
    metadata = {
        "USAF": "421820", "WBAN": "99999", "STATION NAME": "TEST FIXTURE",
        "CTRY": "IN", "LAT": "+28.585", "LON": "+077.206", "ELEV(M)": "+0214.9",
    }
    with (raw_dir / "isd-history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata))
        writer.writeheader()
        writer.writerow(metadata)
    rows = []
    for time, temperature, dewpoint, slp in [
        ("2024-01-01T06:00:00", "+0210,2", "+0250,3", "10025,2"),
        ("2024-01-01T00:00:00", "+0200,1", "+0100,1", "10010,1"),
        ("2024-01-01T03:00:00", "+9999,9", "+0150,1", "99999,9"),
        ("2024-01-01T06:00:00", "+0220,1", "+9999,9", "99999,9"),
        ("2024-01-01T12:30:00", "+0240,1", "+0120,1", "10000,1"),
    ]:
        rows.append({"STATION": SID, "DATE": time, "SOURCE": "4", "NAME": "TEST FIXTURE, IN",
                     "LATITUDE": "28.584511", "LONGITUDE": "77.205783", "ELEVATION": "214.88",
                     "REPORT_TYPE": "FM-12", "QUALITY_CONTROL": "V020", "TMP": temperature,
                     "DEW": dewpoint, "SLP": slp, "REM": 'TEST ONLY, quoted "raw"\nreport'})
    path = raw_dir / "2024" / f"{SID}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return raw_dir, path, rows


def test_native_records_preserve_order_duplicates_raw_attributes_and_lineage(native_files, tmp_path):
    _, path, rows = native_files
    original_bytes = path.read_bytes()
    frame = parse_native_station(path, SID, project_root=tmp_path)
    assert len(frame) == len(rows)
    assert frame["source_row"].tolist() == [1, 2, 3, 4, 5]
    assert frame["raw_DATE"].tolist() == [row["DATE"] for row in rows]
    assert frame["timestamp"].duplicated().sum() == 1
    assert not frame["timestamp"].is_monotonic_increasing
    for column in rows[0]:
        assert frame[f"raw_{column}"].tolist() == [row[column] for row in rows]
    expected_hash = hashlib.sha256(original_bytes).hexdigest()
    assert frame["source_file_sha256"].eq(expected_hash).all()
    assert frame["baseline_record_id"].tolist() == [f"{expected_hash}:{i}" for i in range(1, 6)]
    assert frame["source_file"].eq(f"originals/2024/{SID}.csv").all()
    assert frame["source_retrieved_at_utc"].isna().all()
    assert path.read_bytes() == original_bytes


def test_provider_qc_remains_evidence_not_fault_truth_and_rh_is_unclipped(native_files):
    _, path, _ = native_files
    frame = parse_native_station(path, SID)
    assert frame.loc[0, TEMPERATURE] == 21.0  # Suspect values are retained.
    assert frame.loc[0, f"{TEMPERATURE}__raw_qc"] == "2"
    assert frame.loc[0, "baseline_dewpoint_c__raw_qc"] == "3"
    expected_rh = 100 * np.exp(17.625 * 25 / (243.04 + 25) - 17.625 * 21 / (243.04 + 21))
    assert frame.loc[0, RH] == pytest.approx(expected_rh)
    assert frame.loc[0, RH] > 100  # No silent clipping or source-value repair.
    assert frame["hardware_fault_status"].eq("unknown").all()
    assert not frame["eligible_for_real_training"].any()
    assert not any("label" in column or "anomaly" in column for column in frame)
    assert "relative_humidity_pct" not in frame
    assert "pressure_hpa" not in frame


def test_missing_parent_values_stay_missing_in_derived_and_synthetic_fields(native_files):
    _, path, _ = native_files
    frame = parse_native_station(path, SID)
    assert pd.isna(frame.loc[2, TEMPERATURE])
    assert pd.isna(frame.loc[2, RH])
    assert pd.isna(frame.loc[3, RH])
    for channel in CHANNELS:
        result = simulate_scenario(frame, channel, "stuck", "2024-01-01", 24)
        missing = frame[channel].isna()
        assert result.loc[missing, CHANNELS[channel]].isna().all()
        assert not result.loc[missing, "injected_demo_change"].any()


@pytest.mark.parametrize("scenario", ["baseline", "spike", "drift", "stuck", "dropout"])
def test_scenarios_are_deterministic_preserve_baseline_and_never_mutate_input(native_files, scenario):
    _, path, _ = native_files
    frame = parse_native_station(path, SID)
    before = frame.copy(deep=True)
    result = simulate_scenario(frame, TEMPERATURE, scenario, "2024-01-01T00:00Z", 24, 8)
    repeated = simulate_scenario(frame, TEMPERATURE, scenario, "2024-01-01T00:00Z", 24, 8)
    pd.testing.assert_frame_equal(result, repeated)
    pd.testing.assert_frame_equal(frame, before)
    # is_synthetic_demo is an explicit scenario-wide marker; baseline columns
    # including original provider values and native record IDs remain identical.
    unchanged = [column for column in frame if column != "is_synthetic_demo"]
    pd.testing.assert_frame_equal(result[unchanged], before[unchanged])
    assert result["is_synthetic_demo"].all()
    assert result["scenario_type"].eq(scenario).all()
    assert result["hardware_fault_status"].eq("unknown").all()
    assert not result["eligible_for_real_training"].any()
    assert len(result) == len(frame)
    for other in (RH, PRESSURE):
        np.testing.assert_allclose(result[CHANNELS[other]], frame[other], equal_nan=True)
    if scenario == "baseline":
        assert not result["scenario_applied"].any()
        assert not result["injected_demo_change"].any()


def test_spike_uses_first_chronological_finite_timestamp_including_duplicates(native_files):
    _, path, _ = native_files
    frame = parse_native_station(path, SID)
    result = simulate_scenario(frame, TEMPERATURE, "spike", "2024-01-01T02:00Z", 8, 7)
    assert result["injected_demo_change"].tolist() == [True, False, False, True, False]
    assert result.loc[0, CHANNELS[TEMPERATURE]] == 28
    assert result.loc[3, CHANNELS[TEMPERATURE]] == 29
    assert pd.isna(result.loc[2, CHANNELS[TEMPERATURE]])


def test_drift_uses_elapsed_time_and_stuck_changes_only_different_values(native_files):
    _, path, _ = native_files
    frame = parse_native_station(path, SID)
    drift = simulate_scenario(frame, TEMPERATURE, "drift", "2024-01-01T00:00Z", 12, 12)
    assert drift.loc[0, CHANNELS[TEMPERATURE]] == 27  # Six elapsed hours, not row index.
    assert drift.loc[1, CHANNELS[TEMPERATURE]] == 20
    assert drift.loc[4, CHANNELS[TEMPERATURE]] == 24  # Outside [start, end).
    assert drift["scenario_applied"].tolist() == [True, True, True, True, False]
    assert drift["injected_demo_change"].tolist() == [True, False, False, True, False]
    stuck = simulate_scenario(frame, TEMPERATURE, "stuck", "2024-01-01T00:00Z", 24)
    assert stuck.loc[0, CHANNELS[TEMPERATURE]] == 20  # Chronological first finite report.
    assert stuck["injected_demo_change"].sum() == 3
    assert stuck["scenario_applied"].sum() == 5


def test_dropout_flags_only_values_actually_removed(native_files):
    _, path, _ = native_files
    frame = parse_native_station(path, SID)
    result = simulate_scenario(frame, PRESSURE, "dropout", "2024-01-01T00:00Z", 24)
    assert result[CHANNELS[PRESSURE]].isna().all()
    assert result["injected_demo_change"].tolist() == [True, True, False, False, True]
    assert result["scenario_applied"].all()
    assert frame[PRESSURE].notna().sum() == 3


def test_all_missing_channel_and_empty_interval_are_not_filled(native_files):
    _, path, _ = native_files
    frame = parse_native_station(path, SID)
    frame[PRESSURE] = np.nan
    baseline = simulate_scenario(frame, PRESSURE, "baseline")
    assert baseline[CHANNELS[PRESSURE]].isna().all()
    with pytest.raises(ValueError, match="No finite baseline values"):
        simulate_scenario(frame, PRESSURE, "stuck")
    with pytest.raises(ValueError, match="No finite baseline values"):
        simulate_scenario(frame, TEMPERATURE, "spike", "2024-02-01")
    with pytest.raises(ValueError, match="No native reports"):
        simulate_scenario(frame.iloc[:0], TEMPERATURE)


def test_build_load_and_station_read_validate_hashes_counts_and_policy(native_files, tmp_path):
    raw_dir, _, _ = native_files
    output = tmp_path / "demo"
    manifest = build_demo_bundle(raw_dir, output, station_ids=[SID], project_root=tmp_path)
    assert manifest["dataset_policy"] == DATASET_POLICY
    assert manifest["station_count"] == 1
    assert manifest["row_count"] == 5
    assert manifest["raw_sources"][0]["retrieved_at_utc"] is None
    assert manifest["provider_qc_is_fault_truth"] is False
    assert manifest["channels"][RH]["origin"] == "derived_not_independently_measured"
    bundle = load_demo_bundle(output)
    assert bundle["catalog"].iloc[0]["duplicate_timestamp_rows"] == 1
    assert bundle["catalog"].iloc[0]["sea_level_pressure_present"] == 3
    frame = read_demo_station(bundle, SID)
    assert len(frame) == 5
    parquet = output / manifest["station_files"][0]["path"]
    parquet.write_bytes(parquet.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_demo_station(bundle, SID)


def test_country_and_raw_station_must_match(native_files, tmp_path):
    raw_dir, path, _ = native_files
    with pytest.raises(ValueError, match="raw STATION"):
        parse_native_station(path, "00000099999")
    history_path = raw_dir / "isd-history.csv"
    history = pd.read_csv(history_path, dtype=str)
    history["CTRY"] = "US"
    history.to_csv(history_path, index=False)
    with pytest.raises(ValueError, match="CTRY=IN"):
        build_demo_bundle(raw_dir, tmp_path / "demo", station_ids=[SID])
    assert not (tmp_path / "demo").exists()


def test_native_invalid_dates_fail_without_silent_drops(native_files):
    _, path, _ = native_files
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.loc[2, "DATE"] = "not-a-timestamp"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid DATE.*3"):
        parse_native_station(path, SID)


def test_output_isolation_and_foreign_bundle_rejection(native_files, tmp_path):
    raw_dir, _, _ = native_files
    with pytest.raises(ValueError, match="outside the immutable raw"):
        build_demo_bundle(raw_dir, raw_dir / "demo", station_ids=[SID])
    with pytest.raises(ValueError, match="separate from raw, processed"):
        build_demo_bundle(raw_dir, tmp_path / "data/processed/demo", station_ids=[SID], project_root=tmp_path)
    output = tmp_path / "real_artifacts"
    output.mkdir()
    (output / "detector.json").write_text('{"dataset_policy":"real_observations_only"}')
    with pytest.raises(ValueError, match="not a recognized demo"):
        build_demo_bundle(raw_dir, output, station_ids=[SID])
    (output / "manifest.json").write_text('{"dataset_policy":"real_observations_only"}')
    with pytest.raises(ValueError, match="not an isolated Indian"):
        load_demo_bundle(output)


def test_manifest_paths_cannot_escape_demo_directory(native_files, tmp_path):
    raw_dir, _, _ = native_files
    output = tmp_path / "demo"
    build_demo_bundle(raw_dir, output, station_ids=[SID], project_root=tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["catalog_file"]["path"] = "../originals/isd-history.csv"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside the demo bundle"):
        load_demo_bundle(output)


@pytest.mark.parametrize("kwargs,match", [
    ({"channel": "pressure_hpa"}, "Unknown demo channel"),
    ({"scenario": "real_fault"}, "Unknown scenario"),
    ({"duration_hours": 0}, "greater than zero"),
    ({"magnitude": np.inf}, "finite number"),
    ({"start": "invalid"}, "valid UTC timestamp"),
])
def test_scenario_errors_are_actionable(native_files, kwargs, match):
    _, path, _ = native_files
    frame = parse_native_station(path, SID)
    args = {"channel": TEMPERATURE, **kwargs}
    with pytest.raises(ValueError, match=match):
        simulate_scenario(frame, **args)
