"""CPython JSON-lines adapter and reproducible host measurement for the watchdog.

The portable device module is watchdog.py. This host utility uses only the
standard library; it is not an ESP32 sensor driver or a live data generator.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from urllib.parse import urlparse

try:
    from .watchdog import CHANNELS, VERSION, PreliminaryWatchdog
except ImportError:
    from watchdog import CHANNELS, VERSION, PreliminaryWatchdog


def _invalid_constant(value):
    raise ValueError("nonfinite JSON constant " + value + " is not an observation; use null for missing")


def _stamp(record):
    if "timestamp_s" in record:
        return record["timestamp_s"]
    value = record.get("timestamp")
    if not isinstance(value, str):
        raise ValueError("supply numeric timestamp_s or timezone-aware ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("ISO timestamps must include a timezone")
    return parsed.timestamp()


def _load_policy(args):
    path = getattr(args, "policy", None)
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "edge_frozen_signal_policy_v1" or data.get("comparison") != "strictly_greater":
        raise ValueError("unsupported edge policy schema or threshold comparison")
    if data.get("watchdog_version") != VERSION:
        raise ValueError("policy version does not match this watchdog implementation")
    if data.get("channel_order") != list(CHANNELS):
        raise ValueError("policy channel order does not match the watchdog")
    return data


def _new_watch(args, group, policy):
    if policy is None:
        return PreliminaryWatchdog(group, args.cadence, grace_seconds=args.grace)
    if group not in policy["groups"]:
        raise ValueError("requested group is absent from the exported policy; export it explicitly")
    config = policy["groups"][group]["configuration"]
    if config["cadence_seconds"] != args.cadence or config["grace_seconds"] != args.grace:
        raise ValueError("cadence/grace arguments must match the frozen policy; no silent override")
    return PreliminaryWatchdog(group, **config)


def run_jsonl(args):
    if not args.group:
        raise ValueError("--group station|source is required for an observation stream")
    watch = _new_watch(args, args.group, _load_policy(args))
    source = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    destination = sys.stdout if args.output == "-" else open(args.output, "x", encoding="utf-8")
    try:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_invalid_constant)
                if not isinstance(record, dict):
                    raise ValueError("each JSON line must be an object")
                group = record.get("group")
                if group is None and ("station_id" in record or "source" in record):
                    group = str(record.get("station_id", "")) + "|" + str(record.get("source", ""))
                if group is not None and group != args.group:
                    raise ValueError("record station|source does not match --group")
                kind = record.get("kind", "observation")
                if kind == "heartbeat":
                    result = {"kind": "availability", "result": watch.heartbeat(_stamp(record))}
                elif kind == "observation":
                    result = {"kind": "observation", "original": record,
                              "result": watch.ingest(_stamp(record), record, station_source=group)}
                else:
                    raise ValueError("kind must be observation or heartbeat")
                destination.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
                destination.flush()
            except (ValueError, TypeError) as exc:
                raise ValueError("input line " + str(line_number) + ": " + str(exc)) from exc
    finally:
        if source is not sys.stdin:
            source.close()
        if destination is not sys.stdout:
            destination.close()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deep_size(value, seen=None):
    """CPython reachable object size, not target hardware RAM or peak RSS."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(_deep_size(k, seen) + _deep_size(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return size + sum(_deep_size(v, seen) for v in value)
    if isinstance(value, PreliminaryWatchdog):
        return size + sum(_deep_size(getattr(value, name), seen) for name in value.__slots__)
    return size


def _native_rows(path, expected_day):
    rows, qc_counts, missing = [], {channel: {} for channel in CHANNELS}, {channel: 0 for channel in CHANNELS}
    # Provider positions match the verified SURFRAD adapter. No filling,
    # interpolation or provider-QC-to-hardware-label conversion occurs here.
    fields = ((38, 39), (46, 47), (40, 41))
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 3:
        raise ValueError("SURFRAD file has no native records")
    for line_number, line in enumerate(lines[2:], 3):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 48:
            raise ValueError("expected 48 provider fields at raw line " + str(line_number))
        year, doy, month, day, hour, minute = map(int, parts[:6])
        stamp = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        if stamp.date() != expected_day or stamp.timetuple().tm_yday != doy:
            raise ValueError("raw calendar date disagrees with selected provider day")
        values = {}
        for channel, (value_index, qc_index) in zip(CHANNELS, fields):
            raw_value, raw_qc = parts[value_index], parts[qc_index]
            value, qc = float(raw_value), int(raw_qc)
            if not math.isfinite(value) or not 0 <= qc <= 9:
                raise ValueError("undocumented nonfinite observation or quality code")
            values[channel] = None if value == -9999.9 else value
            missing[channel] += values[channel] is None
            qc_counts[channel][raw_qc] = qc_counts[channel].get(raw_qc, 0) + 1
        rows.append((stamp.timestamp(), values))
    return rows, qc_counts, missing


def measure_originals(args):
    """Measure unchanged, hash-verified daily originals; no accuracy is inferred."""
    if not args.report or not args.day:
        raise ValueError("measurement requires --report and --day YYYY-MM-DD")
    if args.repetitions < 1 or args.repetitions > 100:
        raise ValueError("repetitions must be between 1 and 100")
    report_path = Path(args.report)
    if report_path.exists():
        raise FileExistsError("preserve the existing measurement; choose a new report path")
    archive = Path(args.measure_archive).resolve()
    manifest_path = archive / "surfrad_acquisition.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != "noaa_surfrad":
        raise ValueError("measurement archive must be NOAA SURFRAD")
    day = datetime.strptime(args.day, "%Y-%m-%d").date()
    suffix = day.strftime("%y%j") + ".dat"
    selected = [entry for entry in manifest["files"]
                if entry.get("station_id") in args.stations
                and Path(urlparse(entry["url"]).path).name == entry["station_id"] + suffix]
    if sorted(entry["station_id"] for entry in selected) != sorted(args.stations):
        raise ValueError("the archive must contain exactly one selected daily original per requested station")
    policy = _load_policy(args)
    policy_path = Path(args.policy) if getattr(args, "policy", None) else None
    policy_hash = _sha(policy_path) if policy_path else None
    group_configs = {}
    timing, sources, total_rows, max_state_bytes, max_state_json = [], [], 0, 0, 0
    totals = {"candidate_rows": 0, "physical_domain_alerts": 0, "reporting_range_alerts": 0, "step_alerts": 0, "flatline_alerts": 0}
    for entry in sorted(selected, key=lambda item: item["station_id"]):
        path = (archive / entry["file"]).resolve()
        if not path.is_relative_to(archive):
            raise ValueError("raw source path escapes the archive")
        digest = _sha(path)
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("raw original does not match the pinned acquisition receipt")
        rows, qc_counts, missing = _native_rows(path, day)
        group = entry["station_id"] + "|noaa_surfrad"
        for repeat in range(args.repetitions):
            watch = _new_watch(args, group, policy)
            for timestamp, values in rows:
                started = time.perf_counter_ns()
                result = watch.ingest(timestamp, values)
                timing.append(time.perf_counter_ns() - started)
                if repeat == 0:
                    totals["candidate_rows"] += result["is_candidate"]
                    for alert in result["alerts"]:
                        totals[alert["signal"] + "_alerts"] += 1
                    max_state_bytes = max(max_state_bytes, _deep_size(watch))
                    max_state_json = max(max_state_json, len(json.dumps(watch.snapshot(), allow_nan=False).encode("utf-8")))
        group_configs[group] = {key: watch.snapshot()[key] for key in
                               ("cadence_seconds", "grace_seconds", "step_limits", "flatline_seconds",
                                "flatline_epsilon", "policy_origin", "comparison")}
        if policy:
            group_configs[group]["threshold_group"] = policy["groups"][group]["threshold_group"]
        if _sha(path) != digest:
            raise ValueError("original file changed during measurement")
        total_rows += len(rows)
        sources.append({"group": group, "raw_file": str(path), "raw_file_name": Path(urlparse(entry["url"]).path).name,
                        "sha256": digest, "bytes": entry["bytes"], "source_url": entry["url"],
                        "retrieved_at_utc": entry["retrieved_at_utc"], "rows": len(rows),
                        "raw_rows": {"first": 3, "last": len(rows) + 2}, "provider_qc_counts": qc_counts,
                        "missing_values": missing, "hash_verified_before_and_after": True})
    ordered = sorted(timing)
    mean_ns = sum(timing) / len(timing)
    module = Path(__file__).with_name("watchdog.py")
    if policy_path and _sha(policy_path) != policy_hash:
        raise ValueError("policy changed during measurement")
    report = {
        "schema_version": "edge_watchdog_host_measurement_v2", "watchdog_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "unchanged_real_observations_runtime_only_unknown_hardware_status",
        "source_manifest": {"path": str(manifest_path), "sha256": _sha(manifest_path)}, "sources": sources,
        "scope": {"distinct_original_rows": total_rows, "timed_calls": len(timing), "repetitions": args.repetitions,
                  "selection": "all actual minute rows in the requested daily files; no fill or interpolation",
                  "hardware_fault_labels": "unknown", "accuracy_metrics": None},
        "configuration": {"per_group": group_configs,
                          "policy_file": str(policy_path.resolve()) if policy_path else None,
                          "policy_file_sha256": policy_hash,
                          "policy_semantics": policy["semantics"] if policy else "Default uncalibrated operator thresholds; no fit or selection."},
        "host": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "load_controlled": False},
        "source_size": {"watchdog_py_bytes": module.stat().st_size, "watchdog_py_sha256": _sha(module),
                        "adapter_py_bytes": Path(__file__).stat().st_size, "adapter_py_sha256": _sha(Path(__file__))},
        "state": {"logical_bounds": watch.snapshot()["state_bounds"], "max_observed_host_reachable_bytes": max_state_bytes,
                  "max_observed_snapshot_json_bytes": max_state_json,
                  "measurement": "CPython sys.getsizeof of watchdog and reachable instance state; excludes transient result objects, interpreter and global module code"},
        "runtime": {"mean_us_per_row": mean_ns / 1000, "median_us_per_row": ordered[len(ordered) // 2] / 1000,
                    "p95_us_per_row": ordered[int(.95 * (len(ordered) - 1))] / 1000,
                    "p99_us_per_row": ordered[int(.99 * (len(ordered) - 1))] / 1000,
                    "score_only_rows_per_second": 1e9 / mean_ns,
                    "scope": "host ingest call including result construction; excludes file parsing, JSON serialization, sensor sampling, transport and per-row state-size inspection"},
        "review_evidence_counts_first_pass": totals,
        "hardware_validation": {"esp32_executed": False, "micropython_executed": False,
                                "power_watts": None, "energy_joules": None,
                                "note": "Source compatibility reference only; no device timing, RAM, power, energy, radio or duty-cycle measurements."},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"report": str(report_path), "original_rows": total_rows, "runtime": report["runtime"], "state": report["state"]}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", type=float, required=True, help="explicit expected packet interval in seconds")
    parser.add_argument("--grace", type=float, default=0, help="allowed cadence jitter/heartbeat delay in seconds")
    parser.add_argument("--group", help="one station|source identifier for JSON-lines mode")
    parser.add_argument("--policy", help="optional frozen-signal policy JSON exported by edge/export_policy.py")
    parser.add_argument("--input", default="-", help="existing original JSON-lines file; default stdin")
    parser.add_argument("--output", default="-", help="new result JSON-lines path; default stdout")
    parser.add_argument("--measure-archive", help="host measurement from a pinned SURFRAD archive")
    parser.add_argument("--day", help="explicit original day YYYY-MM-DD for host measurement")
    parser.add_argument("--stations", nargs="+", default=["bon", "fpk", "gwn"])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--report", help="new immutable measurement JSON path")
    args = parser.parse_args(argv)
    try:
        measure_originals(args) if args.measure_archive else run_jsonl(args)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
