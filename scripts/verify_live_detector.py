"""Replay verified immutable originals one row at a time and measure batch parity.

No input scores, artificial faults, labels, interpolation or training are used.
This compatibility/latency exercise is not a new hardware-fault evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from awsad.data.surfrad_native import NativeSurfradArchive
from awsad.live_detector import LiveMinuteDetector, json_safe
from awsad.minute_detection import CHANNELS, SIGNALS, apply_thresholds, compute_signals


def verify(archive_dir, artifact_dir, out, *, stations=("bon", "gwn"), start="2024-10-01T00:00:00Z", minutes=360):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("use a new output directory; preserve prior measurements")
    archive = NativeSurfradArchive(archive_dir)
    engine = LiveMinuteDetector.from_artifacts(artifact_dir, expected_cadence_minutes=1, health_window_rows=360)
    end = pd.Timestamp(start) + pd.Timedelta(minutes=minutes)
    latencies, groups, hashes = [], [], set()
    with (out / "live_scored.jsonl").open("w", encoding="utf-8") as scored_file:
        for station in stations:
            chunks = list(archive.iter_frames(stations=[station], start=start, end=end.isoformat()))
            if not chunks:
                raise ValueError(f"no original rows for {station} in the requested period")
            frame = pd.concat(chunks, ignore_index=True)
            signals, _, predictions, _ = compute_signals(frame, engine.models, engine.config)
            expected = apply_thresholds(frame, signals, predictions, engine.thresholds, engine.config)
            results, elapsed = [], []
            for packet in frame.to_dict("records"):
                before = time.perf_counter()
                result = engine.ingest(packet)
                milliseconds = (time.perf_counter() - before) * 1000
                elapsed.append(milliseconds)
                results.append(result)
                scored_file.write(json.dumps(result, allow_nan=False) + "\n")
            actual = pd.DataFrame(results)
            errors = {}
            for channel in CHANNELS:
                for suffix in [*SIGNALS, "prediction", "prediction_60m", "score"]:
                    key = channel + "__" + suffix
                    a = pd.to_numeric(actual[key]).to_numpy(float)
                    b = expected[key].to_numpy(float)
                    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
                    valid = np.isfinite(a) & np.isfinite(b)
                    errors[key] = float(np.abs(a[valid] - b[valid]).max()) if valid.any() else None
                flags = expected[channel + "__alert"].astype(object)
                assert actual[channel + "__alert"].tolist() == flags.where(flags.notna(), None).tolist()
            assert actual.is_candidate.tolist() == expected.is_candidate.tolist()
            assert actual.reason_codes.tolist() == expected.reason_codes.tolist()
            for channel in CHANNELS:
                np.testing.assert_allclose(pd.to_numeric(actual[channel]).to_numpy(float),
                                           frame[channel].to_numpy(float), rtol=0, atol=0, equal_nan=True)
            assert actual.observation_id.tolist() == frame.observation_id.tolist()
            assert actual.raw_file_sha256.tolist() == frame.raw_file_sha256.tolist()
            hashes.update(frame.raw_file_sha256)
            latencies.extend(elapsed)
            group = f"{station}|noaa_surfrad"
            state = engine.snapshot()["groups"][group]
            groups.append({"group": group, "rows": len(frame), "start": frame.timestamp.iloc[0].isoformat(),
                           "end": frame.timestamp.iloc[-1].isoformat(), "candidate_rows": int(actual.is_candidate.sum()),
                           "unchanged_observed_values": True, "prediction_signal_parity": True,
                           "candidate_and_reason_parity": True, "maximum_absolute_errors": errors,
                           "buffered_rows_at_end": state["buffered_observations"],
                           "mean_inference_ms": float(np.mean(elapsed)), "health": state["health"]})
            print(f"Verified {group}: {len(frame)} original rows; stream/batch parity passed; "
                  f"mean {np.mean(elapsed):.2f} ms/row", flush=True)
    metrics = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "dataset_policy": "unchanged_real_observation_incremental_replay",
               "purpose": "software compatibility and local latency; not fault accuracy or a connected AWS feed",
               "selection_policy": "deterministic fixed October prefix; this previously used period is not an untouched evaluation holdout",
               "input_source": archive.evidence, "raw_files_consumed_and_verified_sha256": sorted(hashes),
               "original_measurements_preserved": True, "precomputed_scores_used_as_input": False,
               "synthetic_observations": 0, "known_hardware_fault_labels": 0,
               "model_info": engine.model_info, "groups": groups, "rows": len(latencies),
               "latency_ms": {"mean": float(np.mean(latencies)), "p50": float(np.quantile(latencies, .5)),
                              "p95": float(np.quantile(latencies, .95)), "p99": float(np.quantile(latencies, .99)),
                              "maximum": float(np.max(latencies))},
               "throughput_rows_per_second": 1000 / float(np.mean(latencies)),
               "latency_scope": "one-row ingest includes frozen model inference, bounded state and JSON-safe record conversion; excludes HTTP/network/source loading",
               "timing_limitation": "Local measurement, not an isolated hardware benchmark; concurrent workloads affect timing.",
               "state_limits": engine.snapshot()["limits"],
               "runtime": {"python": sys.version, "platform": platform.platform()},
               "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                                 for path in ("src/awsad/live_detector.py", "src/awsad/station_health.py", "scripts/verify_live_detector.py")}}
    (out / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(ROOT / "data/raw/surfrad_original"))
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts_minute_20260928"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", default="2024-10-01T00:00:00Z")
    parser.add_argument("--minutes", type=int, default=360)
    parser.add_argument("--stations", nargs="+", default=["bon", "gwn"])
    args = parser.parse_args()
    if not 150 <= args.minutes <= 1440:
        parser.error("minutes must be 150..1440 for a bounded verification run")
    verify(args.archive, args.artifacts, args.out, stations=args.stations, start=args.start, minutes=args.minutes)
