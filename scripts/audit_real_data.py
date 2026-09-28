"""Read-only provenance audit; never generates, fills, or relabels observations.

Uses the standard library so the audit can run without the ML environment.
The header inventory covers every local NOAA station-year CSV. Detailed QC
counts cover only the explicitly listed sample files, never the whole corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {"TMP": "temperature_c", "DEW": "dewpoint_c", "SLP": "sea_level_pressure_hpa"}
SENTINELS = {"TMP": 9999, "DEW": 9999, "SLP": 99999}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_sample(path: Path) -> dict:
    quality = {field: Counter() for field in CHANNELS}
    present = Counter()
    missing = Counter()
    malformed = Counter()
    report_types = Counter()
    station_ids = set()
    times = Counter()
    first = last = None
    rows = 0
    ma1_present = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        for row in reader:
            rows += 1
            station_ids.add(row.get("STATION", ""))
            timestamp = row.get("DATE", "")
            times[timestamp] += 1
            if timestamp:
                first = timestamp if first is None else min(first, timestamp)
                last = timestamp if last is None else max(last, timestamp)
            report_types[row.get("REPORT_TYPE", "")] += 1
            ma1_present += bool(row.get("MA1", "").strip())
            for field in CHANNELS:
                pieces = row.get(field, "").split(",")
                code = pieces[1].strip() if len(pieces) > 1 else ""
                quality[field][code] += 1
                try:
                    raw_value = int(pieces[0])
                except (ValueError, TypeError):
                    if not pieces[0].strip():
                        missing[field] += 1
                    else:
                        malformed[field] += 1
                    continue
                if raw_value == SENTINELS[field]:
                    missing[field] += 1
                else:
                    present[field] += 1
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "station_ids": sorted(station_ids),
        "first_timestamp_as_published": first,
        "last_timestamp_as_published": last,
        "unique_timestamp_strings": len(times),
        "duplicate_timestamp_rows": sum(count - 1 for count in times.values()),
        "blank_timestamps": times.get("", 0),
        "report_type_counts": dict(sorted(report_types.items())),
        "published_columns": header,
        "MA1_nonempty_rows": ma1_present,
        "MA1_note": "Additional pressure field observed; not decoded without its source specification.",
        "present_nonsentinel_counts": dict(present),
        "missing_counts": dict(missing),
        "malformed_counts": dict(malformed),
        "quality_code_counts_all_rows": {key: dict(sorted(value.items())) for key, value in quality.items()},
        "interpretation": "Provider QC information only; no sensor-fault ground truth assigned.",
        "live_origin_reverified": False,
    }


def audit() -> dict:
    raw = ROOT / "data/raw/noaa_isd"
    files = sorted(raw.glob("[0-9][0-9][0-9][0-9]/*.csv"))
    years = Counter(path.parent.name for path in files)
    columns = Counter()
    failures = []
    total_bytes = 0
    for path in files:
        total_bytes += path.stat().st_size
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle))
            columns.update(set(header))
        except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
            failures.append({"path": path.relative_to(ROOT).as_posix(), "error": str(exc)})
    # Explicitly selected local files, without claiming a representative sample.
    sample_ids = ["42182099999", "42027099999", "42809099999"]
    samples = [raw / "2024" / f"{sid}.csv" for sid in sample_ids]
    sample_results = [inspect_sample(path) for path in samples if path.is_file()]
    evidence = []
    for path in sorted((ROOT / "_verify_ghcnh").glob("*")):
        if path.is_file():
            evidence.append({"path": path.relative_to(ROOT).as_posix(),
                             "bytes": path.stat().st_size, "sha256": sha256(path),
                             "verification": "local archived copy; not fetched in this audit"})
    other_sources = {}
    for name in ("openmeteo", "ncpor", "nasa_smap_msl"):
        source_files = [p for p in (ROOT / "data/raw" / name).rglob("*") if p.is_file()]
        other_sources[name] = {"files": len(source_files),
                               "bytes": sum(p.stat().st_size for p in source_files)}
    nab_files = sorted((ROOT / "data/raw/nab/NAB/data").rglob("*.csv"))
    other_sources["nab"] = {"csv_files": len(nab_files),
                             "bytes": sum(p.stat().st_size for p in nab_files)}
    meta_path = ROOT / "data/processed/source_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    return {
        "audit_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "NOT_ELIGIBLE_FOR_REAL_ONLY_HEADLINE_RESULTS",
        "external_verification": "blocked: sandbox socket access denied; browser and elevated-request approval reviews timed out",
        "scope": "all NOAA CSV file headers; detailed records only for the three named 2024 samples that exist",
        "noaa_file_inventory": {
            "station_year_csv_files": len(files),
            "distinct_filename_station_ids": len({path.stem for path in files}),
            "bytes": total_bytes,
            "files_by_year": dict(sorted(years.items())),
            "files_with_each_column": dict(sorted(columns.items())),
            "header_read_failures": failures,
        },
        "noaa_record_samples": sample_results,
        "missing_requested_sample_files": [p.relative_to(ROOT).as_posix() for p in samples if not p.is_file()],
        "other_local_sources": other_sources,
        "existing_processed_metadata": metadata,
        "archived_documentation": evidence,
        "blocking_findings": [
            "Validation/test builders inject manufactured fault values and labels.",
            "Preprocessing fills short gaps by interpolation.",
            "NOAA RH is calculated from TMP/DEW in the existing builder, not a separately observed RH channel.",
            "Existing pressure_hpa is NOAA SLP (sea-level pressure), not station pressure.",
            "QC suspect/erroneous flags do not identify confirmed hardware fault causes.",
            "Rows without a QC flag are not verified fault-free negatives.",
            "Open-Meteo is reanalysis and must not supply primary sensor observations or truth.",
            "Existing processed metadata identifies a legacy snapshot requiring rebuild.",
            "Live dashboard and latency scripts generate random weather observations.",
            "Live origin, licensing and replacement-source samples could not be independently verified this session.",
        ],
        "sensor_fault_ground_truth_available": "not established",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/research/real_data_audit.json")
    args = parser.parse_args()
    result = audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    inventory = result["noaa_file_inventory"]
    print(json.dumps({"result": result["result"],
                      "station_year_files": inventory["station_year_csv_files"],
                      "station_ids": inventory["distinct_filename_station_ids"],
                      "raw_bytes": inventory["bytes"],
                      "sample_files": len(result["noaa_record_samples"]),
                      "report": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
