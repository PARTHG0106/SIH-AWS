"""Synthetic software fixtures only; never observations or benchmark events."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awsad.data import surfrad
from awsad.data import surfrad_native
from awsad.data.surfrad_native import NativeSurfradArchive
from test_surfrad import _file, _record, _test_manifest


def _archive(tmp_path, monkeypatch, records=None):
    root = tmp_path / "software_fixture_archive"
    root.mkdir()
    manifest = _test_manifest(root, monkeypatch)
    manifest["source"] = surfrad.SOURCE
    if records is not None:
        _, receipt = _file(root, records)
        manifest["files"] = [receipt]
    (root / "surfrad_acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def test_native_stream_keeps_all_minutes_raw_values_qc_and_unknown_labels(tmp_path, monkeypatch):
    root, _ = _archive(tmp_path, monkeypatch, [
        _record(), _record(minute=1, humidity="-9999.9", qc="1"),
        _record(minute=3, temperature="22.25", qc="2")])
    archive = NativeSurfradArchive(root)
    frames = list(archive.iter_frames())
    assert len(frames) == 1
    frame = frames[0]
    assert frame["timestamp"].dt.minute.tolist() == [0, 1, 3]
    assert frame["raw_row_number"].tolist() == [3, 4, 5]
    assert frame["label"].isna().all()
    assert pd.isna(frame.loc[1, "relative_humidity_pct"])
    assert frame.loc[1, "relative_humidity_pct__raw_value"] == "-9999.9"
    assert pd.isna(frame.loc[1, "relative_humidity_pct__qc_rejected"])
    assert frame.loc[2, "temperature_c"] == 22.25
    assert frame.loc[2, "temperature_c__qc_rejected"]
    assert frame["pressure_hpa__unit_conversion"].eq("1 mb = 1 hPa").all()
    assert frame["native_averaging_seconds"].eq(60).all()
    report = archive.audit()
    assert report["native_observations"] == 3
    assert report["largest_daily_frame_rows"] == 3
    assert report["known_hardware_fault_labels"] == 0
    assert report["missing_readings"]["relative_humidity_pct"] == 1
    assert report["provider_qc_rejected_present_readings"]["temperature_c"] == 2


def test_native_and_packaged_sources_reconstruct_identical_frames(tmp_path, monkeypatch):
    root, _ = _archive(tmp_path, monkeypatch, [_record(), _record(minute=1)])
    processed = tmp_path / "software_fixture_bundle"
    surfrad.build_surfrad_dataset(root, processed)
    # The native view does not read or depend on an hourly processed table.
    (processed / "observations.parquet").write_bytes(b"untrusted hourly table")
    original, packaged = NativeSurfradArchive(root), NativeSurfradArchive(processed)
    pd.testing.assert_frame_equal(next(original.iter_frames()), next(packaged.iter_frames()))
    assert original.evidence_fingerprint == packaged.evidence_fingerprint


def test_date_filters_are_explicit_utc_and_end_exclusive(tmp_path, monkeypatch):
    root, _ = _archive(tmp_path, monkeypatch, [_record(), _record(minute=1), _record(minute=2)])
    archive = NativeSurfradArchive(root)
    frame = next(archive.iter_frames(start="2024-01-01T05:31:00+05:30",
                                     end="2024-01-01T00:02:00Z"))
    assert frame["timestamp"].dt.minute.tolist() == [1]
    assert frame["raw_row_number"].tolist() == [4]
    assert not list(archive.iter_frames(start="2025-01-01T00:00:00Z"))
    with pytest.raises(ValueError, match="timezone"):
        list(archive.iter_frames(start="2024-01-01"))
    with pytest.raises(ValueError, match="precede"):
        list(archive.iter_frames(start="2024-01-02T00:00Z", end="2024-01-01T00:00Z"))
    with pytest.raises(ValueError, match="stations"):
        list(archive.iter_frames(stations=["fpk"]))


def test_duplicate_off_hour_native_rows_fail_model_admission(tmp_path, monkeypatch):
    root, _ = _archive(tmp_path, monkeypatch, [_record(), _record(minute=1), _record(minute=1)])
    archive = NativeSurfradArchive(root)
    with pytest.raises(ValueError, match="duplicate native"):
        list(archive.iter_frames())


def test_raw_bytes_are_reverified_after_index_construction(tmp_path, monkeypatch):
    root, manifest = _archive(tmp_path, monkeypatch)
    archive = NativeSurfradArchive(root)
    path = root / manifest["files"][0]["file"]
    path.write_bytes(path.read_bytes().replace(b"4.5", b"4.6"))
    with pytest.raises(ValueError, match="immutable acquisition"):
        next(archive.iter_frames())


@pytest.mark.parametrize("tamper", ["unknown_year", "wrong_receipt_year", "missing_day", "changed_docs", "failed_download"])
def test_unverified_source_evidence_fails_closed(tmp_path, monkeypatch, tamper):
    root, manifest = _archive(tmp_path, monkeypatch)
    if tamper == "unknown_year":
        manifest["years"] = [1997]
    elif tamper == "wrong_receipt_year":
        manifest["files"][0]["year"] = 2023
    elif tamper == "missing_day":
        manifest["files"] = []
    elif tamper == "changed_docs":
        manifest["documents"]["corrections"]["sha256"] = "0" * 64
    else:
        manifest["failures"] = [{"url": "test-only", "error": "test-only"}]
    (root / "surfrad_acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        NativeSurfradArchive(root)


def test_filtered_view_cannot_hide_invalid_off_hour_raw_records(tmp_path, monkeypatch):
    root, _ = _archive(tmp_path, monkeypatch, [_record(), _record(minute=1, temperature="nan")])
    with pytest.raises(ValueError, match="nonfinite"):
        next(NativeSurfradArchive(root).iter_frames(end="2024-01-01T00:01:00Z"))


def test_declared_bounded_acquisition_checks_filtered_complete_inventory(tmp_path, monkeypatch):
    root, manifest = _archive(tmp_path, monkeypatch)
    receipt = manifest["inventories"][0]
    listing = root / receipt["file"]
    listing.write_text(listing.read_text() + '<a href="bon24002.dat">bon24002.dat</a>')
    receipt["sha256"] = hashlib.sha256(listing.read_bytes()).hexdigest()
    receipt["bytes"] = listing.stat().st_size
    manifest["date_range"] = {"start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"}
    (root / "surfrad_acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = NativeSurfradArchive(root)
    assert len(archive.entries) == 1
    assert archive.scope["acquisition_date_range"]["end"] == "2024-01-02T00:00:00+00:00"
    assert archive.audit()["native_observations"] == 2
    # The same files cannot masquerade as a complete two-day acquisition.
    manifest["date_range"]["end"] = "2024-01-03T00:00:00Z"
    (root / "surfrad_acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        NativeSurfradArchive(root)


@pytest.mark.parametrize("start,end", [
    ("2024-01-01", "2024-02-01T00:00:00Z"),
    ("2024-01-01T00:01:00Z", "2024-02-01T00:00:00Z"),
    ("2024-02-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    ("1997-01-01T00:00:00Z", "1997-02-01T00:00:00Z"),
])
def test_invalid_acquisition_scope_fails_before_network(tmp_path, monkeypatch, start, end):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid scope must not start acquisition")
    monkeypatch.setattr(surfrad_native, "_acquire_retry", forbidden)
    with pytest.raises(ValueError):
        surfrad_native.fetch_surfrad_date_range(tmp_path, start=start, end=end)


def test_bounded_fetch_downloads_only_provider_listed_days_in_declared_range(tmp_path, monkeypatch):
    acquired_urls = []
    base = surfrad_native.BASE
    year_url = base + "Bondville_IL/2025/"
    def document(url, folder):
        return {"url": url, "file": "test-only-doc", "sha256": "test-only-hash"}
    def listing(url, folder):
        links = {base: ["Bondville_IL/"], base + "Bondville_IL/": ["2025/"],
                 year_url: ["bon25001.dat", "bon25003.dat", "bon25031.dat", "bon25032.dat"]}[url]
        return document(url, folder), links
    def daily(url, folder, **kwargs):
        acquired_urls.append(url)
        return document(url, folder)
    monkeypatch.setattr(surfrad_native, "_acquire_retry", document)
    monkeypatch.setattr(surfrad_native, "_listed", listing)
    monkeypatch.setattr(surfrad_native, "acquire", daily)
    out = tmp_path / "software_fixture_acquisition"
    manifest = surfrad_native.fetch_surfrad_date_range(out, stations=["bon"],
        start="2025-01-01T00:00:00Z", end="2025-02-01T00:00:00Z")
    assert sorted(acquired_urls) == [year_url + name for name in ("bon25001.dat", "bon25003.dat", "bon25031.dat")]
    assert manifest["failures"] == []
    assert manifest["years"] == [2025]
    assert manifest["inventories"][-1]["listed_daily_files"] == 4
    assert manifest["inventories"][-1]["selected_daily_files"] == 3
    assert manifest["date_range"]["end"] == "2025-02-01T00:00:00+00:00"
    with pytest.raises(FileExistsError):
        surfrad_native.fetch_surfrad_date_range(out, stations=["bon"],
            start="2025-01-01T00:00:00Z", end="2025-02-01T00:00:00Z")
