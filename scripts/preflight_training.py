"""Read-only training admission check; never generates observations or labels.

Exit 0 means verified observations are eligible, 2 means blocked. Registered
SURFRAD inputs undergo complete raw-to-processed verification; legacy datasets
return 2 and an actionable JSON report. Optional repository
inventory inspects native NOAA headers and ZIP indexes, without extracting data
or loading processed Parquet arrays. Timestamps are filesystem metadata, not
proof of download time, measurement origin, or new observations.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awsad.data.training_contract import inspect_training_data, write_training_report


def _file_summary(paths: list[Path], since: datetime) -> dict:
    stats = [(path, path.stat()) for path in paths]
    newest = max(stats, key=lambda item: item[1].st_mtime) if stats else None
    recent = [(path, stat) for path, stat in stats if stat.st_mtime >= since.timestamp()]
    return {
        "files": len(stats),
        "bytes": sum(stat.st_size for _, stat in stats),
        "files_modified_since_audit": len(recent),
        "latest_modified_at_utc": datetime.fromtimestamp(newest[1].st_mtime, timezone.utc).isoformat() if newest else None,
        "latest_modified_file": str(newest[0]) if newest else None,
    }


def inventory_local_data(repository_root: Path) -> dict:
    """Inventory actual local files, keeping source-code bundles distinct."""
    root = repository_root.resolve()
    data = root / "data"
    since = datetime(2026, 9, 23, tzinfo=timezone.utc)
    result = {
        "scope": "data/raw, data/interim, data/processed, Kaggle ZIP indexes; no network or dataset generation",
        "does_not_establish_remote_dataset_freshness": True,
        "filesystem_timestamp_cutoff_utc": since.isoformat(),
        "timestamps_are_acquisition_evidence": False,
        "raw_sources": {},
        "noaa_native_headers": {},
        "processed_candidates": [],
        "archives": [],
        "errors": [],
    }
    raw = data / "raw"
    for source in sorted(raw.iterdir()) if raw.is_dir() else []:
        if not source.is_dir():
            continue
        # NAB includes code, result tables, and test data. Count only its data
        # collection here, without implying all files are real weather data.
        source_data = source / "NAB" / "data" if source.name == "nab" else source
        try:
            files = [path for path in source_data.rglob("*") if path.is_file()]
            result["raw_sources"][source.name] = _file_summary(files, since)
            result["raw_sources"][source.name]["path"] = str(source_data)
        except OSError as exc:
            result["errors"].append({"path": str(source_data), "error": str(exc)})

    noaa_files = sorted((raw / "noaa_isd").glob("[0-9][0-9][0-9][0-9]/*.csv"))
    headers = Counter()
    failures = []
    for path in noaa_files:
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                headers.update(set(next(csv.reader(handle))))
        except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
            failures.append({"path": str(path), "error": str(exc)})
    result["noaa_native_headers"] = {
        **_file_summary(noaa_files, since),
        "distinct_filename_station_ids": len({path.stem for path in noaa_files}),
        "files_by_year": dict(sorted(Counter(path.parent.name for path in noaa_files).items())),
        "files_with_TMP": headers["TMP"],
        "files_with_DEW": headers["DEW"],
        "files_with_SLP": headers["SLP"],
        "files_with_MA1": headers["MA1"],
        "RH_prefixed_headers": {name: count for name, count in sorted(headers.items()) if name.upper().startswith("RH")},
        "header_read_failures": failures,
        "interpretation": "Header/file inventory only. Column presence does not establish nonmissing observations, independent measured RH, station pressure, instrument type, fault truth, or eligible training rows.",
    }
    for directory in (data / "processed", data / "interim"):
        if directory.is_dir():
            files = [path for path in directory.rglob("*") if path.is_file()]
            result["processed_candidates"].append({"path": str(directory), **_file_summary(files, since)})

    for directory in (data / "kaggle_upload", data / "kaggle_staging"):
        for path in sorted(directory.glob("*.zip")):
            archive = {"path": str(path), **_file_summary([path], since)}
            try:
                with zipfile.ZipFile(path) as zipped:
                    entries = zipped.infolist()
                    archive["entries"] = len(entries)
                    archive["processed_parquet_entries"] = [entry.filename for entry in entries if "/processed/" in f"/{entry.filename}" and entry.filename.endswith(".parquet")]
                    manifests = []
                    for entry in entries:
                        if Path(entry.filename).name not in {"MANIFEST.json", "source_metadata.json", "training_contract.json"}:
                            continue
                        if entry.file_size > 2 * 1024 * 1024:
                            manifests.append({"entry": entry.filename, "error": "metadata too large"})
                            continue
                        try:
                            payload = json.loads(zipped.read(entry).decode("utf-8-sig"))
                            if not isinstance(payload, dict):
                                raise ValueError("metadata is not an object")
                            manifests.append({"entry": entry.filename, **{key: payload[key] for key in (
                                "status", "parser_version", "bundle_role", "processed_data_included", "source_roles", "rh_policy", "pressure_policy",
                            ) if key in payload}})
                        except (UnicodeError, ValueError) as exc:
                            manifests.append({"entry": entry.filename, "error": str(exc)})
                    archive["metadata"] = manifests
            except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
                archive["error"] = str(exc)
            result["archives"].append(archive)
    return result


def inspect_builder_log(path: Path) -> dict:
    """Record supplied run evidence without treating it as remote-file proof."""
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Builder log is too large for this bounded metadata inspection")
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    result = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "evidence_kind": "user_supplied_builder_run_log",
        "verification_scope": "Log content inspected locally. Remote raw files, processed arrays, and manifest contents were not downloaded or verified by this check.",
        "reported": {},
        "evidence_lines": {},
        "interpretation": [
            "A newer remote builder run can exist while the repository's local processed files remain older; local timestamps do not determine remote freshness.",
            "The intended workflow is source-only dataset -> online data builder -> processed dataset -> training notebook. A source-only ZIP is expected at the first stage.",
            "Provider suspect/erroneous QC counts are quality evidence, not independently confirmed hardware-fault labels; derived RH does not establish a measured humidity sensor.",
            "An observed newer build does not by itself establish real-only eligibility; injected observations and labels remain ineligible under the current policy.",
        ],
    }
    markers = {
        "source_manifest_sha256": r"source manifest sha256:\s*([0-9a-fA-F]{64})",
        "build_id": r"build id:\s*([^\s|]+)",
        "build_utc": r"build UTC:\s*(\S+)",
        "parser_version": r"ISD parser:\s*(\S+)",
        "fault_injector_version": r"fault injector:\s*(\S+)",
    }
    for number, line in enumerate(lines, start=1):
        for key, pattern in markers.items():
            match = re.search(pattern, line)
            if match:
                result["reported"][key] = match.group(1)
                result["evidence_lines"][key] = number
        if "label provenance:" in line:
            payload = ast.literal_eval(line.split("label provenance:", 1)[1].strip())
            if not isinstance(payload, dict):
                raise ValueError("Builder label-provenance log entry is not a mapping")
            result["reported"]["label_provenance"] = payload
            result["evidence_lines"]["label_provenance"] = number
        if "processed written:" in line:
            result["reported"]["processed_written_line"] = line.strip()
            result["evidence_lines"]["processed_written_line"] = number
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", "--out", type=Path, default=ROOT / "artifacts" / "training_preflight.json")
    parser.add_argument("--inventory-root", type=Path, help="Optionally inventory this repository's existing data and ZIP metadata.")
    parser.add_argument("--builder-log", type=Path, help="Record supplied online-builder log claims separately from local file verification.")
    args = parser.parse_args(argv)
    report = inspect_training_data(args.processed_dir)
    if args.inventory_root is not None:
        report["local_inventory"] = inventory_local_data(args.inventory_root)
    if args.builder_log is not None:
        report["builder_run_log"] = inspect_builder_log(args.builder_log)
    write_training_report(report, args.output)
    print(json.dumps({"status": report["status"], "eligible": report["eligible"],
                      "blockers": report["blockers"], "report": str(args.output.resolve())}, indent=2))
    return 0 if report["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
