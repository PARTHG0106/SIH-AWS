"""Synthetic SOFTWARE fixtures confined to tests, never source observations."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awsad.data import surfrad
from awsad.data.fetch_surfrad import DOCUMENTS


def _record(hour=0, minute=0, *, temperature="4.5", humidity="80.1", pressure="999.2", qc="0"):
    row = ["2024", "1", "1", "1", str(hour), str(minute), "0.000", "155.00"]
    row += [value for _ in range(20) for value in ("0.0", "0")]
    row[38:42] = [temperature, qc, humidity, qc]
    row[46:48] = [pressure, qc]
    assert len(row) == 48
    return " ".join(row)


def _file(tmp_path, records):
    # Header layout was verified in acquired bon24001.dat: six fields with
    # literal "m version" markers, not four bare metadata values.
    raw = ("Bondville, IL\n40.0519 -88.3731 213 m version 1\n" + "\n".join(records) + "\n").encode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    path = tmp_path / (digest + ".dat")
    path.write_bytes(raw)
    receipt = {"file": path.name, "sha256": digest, "bytes": len(raw),
               "url": "https://gml.noaa.gov/aftp/data/radiation/surfrad/Bondville_IL/2024/bon24001.dat",
               "retrieved_at_utc": "2026-09-26T03:00:00+00:00", "station_id": "bon", "year": 2024}
    receipt["final_url"] = receipt["url"]
    return path, receipt


def test_exact_hour_keeps_actual_rows_and_field_lineage(tmp_path):
    path, receipt = _file(tmp_path, [_record(), _record(minute=1), _record(hour=2)])
    frame = surfrad.parse_surfrad_daily(path, receipt)
    assert len(frame) == 2
    assert frame["timestamp"].dt.hour.tolist() == [0, 2]  # missing hour1 is NOT inserted
    assert frame["raw_row_number"].tolist() == [3, 5]
    assert frame["observation_id"].tolist() == [receipt["sha256"] + ":3", receipt["sha256"] + ":5"]
    assert frame["temperature_c"].tolist() == [4.5, 4.5]
    assert frame["pressure_hpa"].tolist() == [999.2, 999.2]
    assert frame["relative_humidity_pct__raw_value"].tolist() == ["80.1", "80.1"]
    assert frame["pressure_hpa__raw_unit"].eq("mb").all()
    assert frame["pressure_hpa__unit_conversion"].eq("1 mb = 1 hPa").all()
    assert frame["native_file_version"].eq("1").all()
    assert frame["label"].isna().all()
    assert str(frame["label"].dtype) == "Int8"
    assert str(frame["timestamp"].dt.tz) == "UTC"


def test_missing_reading_and_provider_flags_are_not_fault_labels(tmp_path):
    path, receipt = _file(tmp_path, [_record(humidity="-9999.9", qc="1")])
    row = surfrad.parse_surfrad_daily(path, receipt).iloc[0]
    assert pd.isna(row["relative_humidity_pct"])
    assert row["relative_humidity_pct__raw_value"] == "-9999.9"
    assert row["relative_humidity_pct__raw_qc"] == "1"
    assert pd.isna(row["relative_humidity_pct__qc_accepted"])
    assert pd.isna(row["relative_humidity_pct__qc_rejected"])
    assert row["temperature_c__qc_rejected"]
    assert not row["temperature_c__qc_accepted"]
    assert pd.isna(row["label"])


def test_native_rows_and_duplicate_reports_are_preserved(tmp_path):
    path, receipt = _file(tmp_path, [_record(), _record(), _record(minute=1)])
    frame = surfrad.parse_surfrad_daily(path, receipt, selection="native")
    assert len(frame) == 3 and frame["timestamp"].duplicated().sum() == 1
    assert frame["observation_id"].is_unique


def test_no_minute_zero_does_not_create_a_measurement(tmp_path):
    path, receipt = _file(tmp_path, [_record(minute=1)])
    assert surfrad.parse_surfrad_daily(path, receipt).empty


def test_modified_raw_bytes_fail_before_parsing(tmp_path):
    path, receipt = _file(tmp_path, [_record()])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable"):
        surfrad.parse_surfrad_daily(path, receipt)


@pytest.mark.parametrize("change", ["schema", "timestamp", "station", "nonfinite"])
def test_malformed_or_unverified_records_fail_closed(tmp_path, change):
    record = _record()
    if change == "schema":
        record += " 9"
    elif change == "timestamp":
        record = record.replace("2024 1 1 1", "2024 2 1 1")
    elif change == "nonfinite":
        record = _record(temperature="NaN")
    path, receipt = _file(tmp_path, [record])
    if change == "station":
        receipt["url"] = receipt["url"].replace("bon24001", "gwn24001")
    with pytest.raises(ValueError):
        surfrad.parse_surfrad_daily(path, receipt)


def test_unreviewed_year_and_wrong_station_are_rejected(tmp_path):
    path, receipt = _file(tmp_path, [_record()])
    with pytest.raises(ValueError, match="scope"):
        surfrad.parse_surfrad_daily(path, receipt, station_code="gwn")
    receipt["url"] = receipt["url"].replace("2024", "2017").replace("bon24", "bon17")
    with pytest.raises(ValueError, match="scope"):
        surfrad.parse_surfrad_daily(path, receipt)


def _test_manifest(tmp_path, monkeypatch):
    path, receipt = _file(tmp_path, [_record(), _record(hour=1, humidity="-9999.9", qc="1")])
    documents, hashes = {}, {}
    for role, alias in surfrad.DOCUMENT_ALIASES.items():
        # Deliberately fake documents, admitted only through a test-local patch.
        # The production module pins actual independently inspected NOAA hashes.
        doc = tmp_path / (role + ".txt")
        doc.write_text("TEST FIXTURE ONLY " + role, encoding="utf-8")
        digest = hashlib.sha256(doc.read_bytes()).hexdigest()
        hashes[role] = digest
        documents[alias] = {"file": doc.name, "sha256": digest, "url": DOCUMENTS[alias],
                            "final_url": DOCUMENTS[alias], "retrieved_at_utc": "2026-09-26T03:00:00+00:00",
                            "bytes": doc.stat().st_size}
    monkeypatch.setattr(surfrad, "REVIEWED_DOCUMENTS", hashes)
    listing = tmp_path / "test_only_inventory.html"
    listing.write_text('<a href="bon24001.dat">bon24001.dat</a>', encoding="utf-8")
    directory = "https://gml.noaa.gov/aftp/data/radiation/surfrad/Bondville_IL/2024/"
    inventory_receipt = {"file": listing.name, "sha256": hashlib.sha256(listing.read_bytes()).hexdigest(),
                         "bytes": listing.stat().st_size, "url": directory, "final_url": directory,
                         "retrieved_at_utc": "2026-09-26T03:00:00+00:00"}
    manifest = {"stations": ["bon"], "years": [2024], "documents": documents,
                "inventories": [inventory_receipt], "files": [receipt], "failures": [],
                "completed_at_utc": "2026-09-26T03:00:00+00:00"}
    (tmp_path / "surfrad_acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_builder_preserves_unknowns_and_packages_replayable_contract(tmp_path, monkeypatch):
    archive = tmp_path / "software_fixture_archive"
    archive.mkdir()
    manifest = _test_manifest(archive, monkeypatch)
    out = tmp_path / "test_bundle"
    metadata = surfrad.build_surfrad_dataset(archive, out)
    frame = pd.read_parquet(out / "observations.parquet")
    assert len(frame) == 2 and frame["label"].isna().all()
    assert frame["relative_humidity_pct"].isna().sum() == 1
    assert metadata["qc_agreement"]["documented"]
    raw = json.loads((out / "raw_file_manifest.json").read_text())["files"][0]
    replay = surfrad.parse_surfrad_daily(out / raw["path"], raw)
    pd.testing.assert_frame_equal(frame, replay)
    contract = json.loads((out / "training_contract.json").read_text())
    assert contract["adapter"] == "noaa_surfrad_v1"
    assert contract["labels"]["unknown"] == "preserved"
    assert not contract["labels"]["provider_qc_is_fault_truth"]
    assert not (out / "train_clean.parquet").exists()
    for ref in contract["evidence"].values():
        assert hashlib.sha256((out / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    from awsad.data.training_contract import require_eligible_training_data
    report = require_eligible_training_data(out)
    assert report["eligible"]
    assert report["verification"]["verified_selected_observations"] == 2
    assert report["verification"]["known_fault_labels"] == 0
    with pytest.raises(FileExistsError):
        surfrad.build_surfrad_dataset(archive, out)


def test_incomplete_acquisition_does_not_create_dataset(tmp_path, monkeypatch):
    manifest = _test_manifest(tmp_path, monkeypatch)
    manifest["failures"] = [{"url": "test-only", "error": "test-only error"}]
    (tmp_path / "surfrad_acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "never_built"
    with pytest.raises(ValueError, match="incomplete"):
        surfrad.build_surfrad_dataset(tmp_path, out)
    assert not out.exists()


def _rehash_test_bundle(out):
    """Simulate attacker/upstream bug updating metadata after fabricating data."""
    processed_path = out / "processed_file_manifest.json"
    processed = json.loads(processed_path.read_text())
    for item in processed["files"]:
        item["sha256"] = hashlib.sha256((out / item["path"]).read_bytes()).hexdigest()
    processed_path.write_text(json.dumps(processed), encoding="utf-8")
    contract_path = out / "training_contract.json"
    contract = json.loads(contract_path.read_text())
    for item in contract["evidence"].values():
        item["sha256"] = hashlib.sha256((out / item["path"]).read_bytes()).hexdigest()
    contract_path.write_text(json.dumps(contract), encoding="utf-8")


@pytest.mark.parametrize("tamper", ["observation", "label", "raw", "inventory_subset"])
def test_independent_verifier_rejects_tampering_even_with_updated_hashes(tmp_path, monkeypatch, tamper):
    archive = tmp_path / "software_fixture_archive"
    archive.mkdir()
    _test_manifest(archive, monkeypatch)
    out = tmp_path / "test_bundle"
    surfrad.build_surfrad_dataset(archive, out)
    if tamper in {"observation", "label"}:
        frame = pd.read_parquet(out / "observations.parquet")
        if tamper == "observation":
            frame.loc[0, "temperature_c"] = 22.25
        else:
            frame.loc[0, "label"] = 0  # unreviewed cannot become normal
        frame.to_parquet(out / "observations.parquet", index=False)
    elif tamper == "raw":
        raw = next((out / "raw").glob("*.dat"))
        raw.write_bytes(raw.read_bytes() + b" ")
    else:
        acquired_path = out / "acquisition_manifest.json"
        acquired = json.loads(acquired_path.read_text())
        listing = out / acquired["inventories"][0]["path"]
        listing.write_text(listing.read_text() + '<a href="bon24002.dat">bon24002.dat</a>', encoding="utf-8")
        acquired["inventories"][0]["sha256"] = hashlib.sha256(listing.read_bytes()).hexdigest()
        acquired["inventories"][0]["bytes"] = listing.stat().st_size
        acquired_path.write_text(json.dumps(acquired), encoding="utf-8")
    _rehash_test_bundle(out)
    from awsad.data.training_contract import require_eligible_training_data, TrainingDataContractError
    with pytest.raises(TrainingDataContractError):
        require_eligible_training_data(out)
