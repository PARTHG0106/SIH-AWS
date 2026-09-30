"""Evaluate reviewed intervals against immutable native-minute replay artifacts.

Usage: python scripts/evaluate_real_events.py --artifacts DIR --reviews CSV --out NEW.json
Optional --signal evaluates one saved signal with its frozen threshold, allowing
the same review set to compare the combined detector with simple baselines.
This tool never trains, tunes thresholds, assigns reviews or modifies artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from awsad.evaluation.real_events import _decisions, evaluate_reviewed_events, validate_reviews

CHANNELS = ("temperature_c", "pressure_hpa", "relative_humidity_pct")
SIGNALS = ("forecast_residual", "abrupt_change", "flatline_minutes", "hour_residual", "sustained_deviation")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_artifacts(artifacts: Path, reviews_path: Path, *, signal: str | None = None):
    """Read one shard at a time; retain only independently reviewed decisions.

    All decisions contribute to exact unknown/coverage counts. Review intervals
    spanning shards are joined before evaluation, so month boundaries do not
    split a reviewed event or a false-alert episode.
    """
    if signal is not None and signal not in SIGNALS:
        raise ValueError("Unsupported detector signal")
    artifacts, reviews_path = Path(artifacts).resolve(), Path(reviews_path).resolve()
    provenance_path = artifacts / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    shards = provenance.get("scored_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Artifact provenance must list scored_shards")
    reviews = validate_reviews(pd.read_csv(reviews_path, keep_default_na=False))
    accepted = reviews.loc[reviews.review_status.ne("unknown")]
    intervals = {key: list(zip(group.start_utc, group.end_utc))
                 for key, group in accepted.groupby(["group", "channel"], sort=False)}
    detector_path = artifacts / "detector.json"
    detector = json.loads(detector_path.read_text(encoding="utf-8")) if detector_path.exists() else None
    if signal and detector is None:
        raise ValueError("Baseline evaluation requires frozen detector.json thresholds")
    selected = []
    boundaries = defaultdict(list)
    all_rows = all_alerts = native_rows = 0
    split_counts = Counter()
    input_files = []
    names = set()
    for entry in shards:
        path = (artifacts / entry["file"]).resolve()
        if not path.is_relative_to(artifacts) or path in names:
            raise ValueError("Scored shard paths must be unique and stay within artifacts")
        names.add(path)
        digest = sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"Scored shard bytes changed: {entry['file']}")
        cols = ["group", "timestamp", "observation_id", "split"]
        cols += [c + "__" + (signal if signal else "alert") for c in CHANNELS]
        if signal:
            cols.append("threshold_group")
        frame = pd.read_parquet(path, columns=cols)
        if frame.empty or len(frame) != entry["rows"] or frame.group.nunique() != 1:
            raise ValueError("Scored shard rows/group do not match replay contract")
        group_name = frame.group.iloc[0]
        if group_name != entry["group"]:
            raise ValueError("Scored shard group differs from provenance")
        native_rows += len(frame)
        split_counts.update(frame.split.astype(str))
        for channel in CHANNELS:
            block = frame[["group", "timestamp", "observation_id", "split"]].copy()
            block["channel"] = channel
            if signal:
                values = pd.to_numeric(frame[channel + "__" + signal], errors="raise").to_numpy(float)
                threshold_values = {}
                for name in frame.threshold_group.unique():
                    record = detector["thresholds"][name][channel][signal]
                    if record.get("comparison") != "strictly_greater":
                        raise ValueError("Unsupported saved threshold comparison")
                    threshold_values[name] = record["threshold"]
                thresholds = frame.threshold_group.map(threshold_values).to_numpy(float)
                if not np.isfinite(thresholds).all() or (thresholds < 0).any():
                    raise ValueError("Invalid saved detector threshold")
                block["is_alert"] = pd.array(
                    np.where(np.isfinite(values), values > thresholds, None), dtype="boolean")
            else:
                block["is_alert"] = frame[channel + "__alert"]
            block = _decisions(block)
            if channel == CHANNELS[0]:
                start, end = block.timestamp.iloc[0], block.timestamp.iloc[-1]
                if start != pd.Timestamp(entry["start"]) or end != pd.Timestamp(entry["end"]):
                    raise ValueError("Scored shard timestamp bounds differ from provenance")
                boundaries[group_name].append((start, end))
            all_rows += len(block)
            all_alerts += int(block.is_alert.fillna(False).sum())
            mask = np.zeros(len(block), dtype=bool)
            for start, end in intervals.get((group_name, channel), []):
                mask |= (block.timestamp.ge(start) & block.timestamp.lt(end)).to_numpy()
            if mask.any():
                selected.append(block.loc[mask])
        input_files.append({"file": entry["file"], "sha256": digest, "rows": len(frame)})
    for group, spans in boundaries.items():
        ordered = sorted(spans)
        if any(next_start <= end for (_, end), (next_start, _) in zip(ordered, ordered[1:])):
            raise ValueError(f"Scored shards overlap in group {group}")
    decision_columns = ["group", "timestamp", "observation_id", "split", "channel", "is_alert"]
    reviewed_decisions = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=decision_columns)
    report = evaluate_reviewed_events(reviewed_decisions, reviews, expected_cadence_seconds=60)
    report.update(input_observation_rows=all_rows,
                  unknown_observation_rows=all_rows - len(reviewed_decisions),
                  unknown_alert_rows=all_alerts - int(reviewed_decisions.is_alert.fillna(False).sum()))
    report["input_native_observation_rows"] = native_rows
    report["decision_row_semantics"] = "One row per native observation/channel; three channels per native observation"
    report["native_rows_by_split"] = dict(sorted(split_counts.items()))
    report["evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["detector_decision"] = "saved_combined_channel_alert" if signal is None else "saved_signal_" + signal
    report["thresholds_retuned"] = False
    report["review_validation_note"] = "Checks declarations/intervals only; the named reviewer must verify independent evidence content."
    report["inputs"] = {"artifact_directory": str(artifacts), "reviews_csv": str(reviews_path),
                        "reviews_sha256": sha256(reviews_path), "provenance_sha256": sha256(provenance_path),
                        "detector_sha256": sha256(detector_path) if detector_path.exists() else None,
                        "scored_shards": input_files,
                        "evaluation_source_sha256": sha256(ROOT / "src/awsad/evaluation/real_events.py"),
                        "cli_source_sha256": sha256(Path(__file__))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--signal", choices=SIGNALS)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve existing evaluation; choose a new --out path")
    report = evaluate_artifacts(args.artifacts, args.reviews, signal=args.signal)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "confirmed_event_count", "event_recall",
                      "reviewed_background_intervals", "unknown_observation_rows")}, indent=2))


if __name__ == "__main__":
    main()
