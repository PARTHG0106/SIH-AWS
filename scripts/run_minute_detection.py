"""Fit/calibrate native-minute detectors and export a real-observation replay.

Development ends before the previously examined November-December 2024 test.
Use --replay-only with a frozen detector for a separately acquired fresh period.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from awsad.data.acquisition import sha256_file
from awsad.data.surfrad_native import NativeSurfradArchive
from awsad.minute_detection import (BASE_COLUMNS, CHANNELS, SIGNALS, MinuteConfig,
    apply_thresholds, causal_features, compute_signals, config_fingerprint,
    fit_forecasters, fit_thresholds, flatline_durations, split_names, write_json)


def build_cache(archive_dir, cache_dir, *, start=None, end=None):
    """Materialize full native provenance once; never load an entire station year."""
    archive = NativeSurfradArchive(archive_dir)
    cache = Path(cache_dir)
    manifest_path = cache / "native_manifest.json"
    identity = {"source_fingerprint": archive.evidence_fingerprint, "start": start, "end": end,
                "parser_sha256": sha256_file(ROOT / "src/awsad/data/surfrad.py")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        recorded_identity = dict(manifest["identity"])
        # The creator loader hash is provenance, not a requirement that additive
        # acquisition features force unchanged native observations to be rebuilt.
        recorded_identity.pop("native_loader_sha256", None)
        if recorded_identity != identity:
            raise ValueError("cache does not match the exact source/parser/view; use a new directory")
        for entry in manifest["shards"]:
            if sha256_file(cache / entry["file"]) != entry["sha256"]:
                raise ValueError("native cache bytes changed")
        return manifest
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError("incomplete native cache; preserve it and use a new directory")
    cache.mkdir(parents=True, exist_ok=True)
    shards, frames, key = [], [], None

    def flush():
        if not frames:
            return
        frame = pd.concat(frames, ignore_index=True)
        name = key + ".parquet"
        frame.to_parquet(cache / name, index=False, compression="zstd")
        shards.append({"file": name, "sha256": sha256_file(cache / name), "rows": len(frame),
                       "station_id": str(frame.station_id.iloc[0]), "source": str(frame.source.iloc[0]),
                       "start": frame.timestamp.iloc[0].isoformat(), "end": frame.timestamp.iloc[-1].isoformat(),
                       "raw_files": int(frame.raw_file_sha256.nunique()),
                       "missing": {c: int(frame[c].isna().sum()) for c in CHANNELS}})
        print(f"Verified native {key}: {len(frame):,} records", flush=True)

    for frame in archive.iter_frames(start=start, end=end):
        new_key = frame.station_id.iloc[0] + "_" + frame.timestamp.iloc[0].strftime("%Y_%m")
        if key != new_key:
            flush()
            frames, key = [], new_key
        frames.append(frame)
    flush()
    if not shards:
        raise ValueError("no original native observations in requested view")
    manifest = {"dataset_policy": "real_observations_only", "identity": identity,
                "creator_native_loader_sha256": sha256_file(ROOT / "src/awsad/data/surfrad_native.py"),
                "evidence": archive.evidence, "scope": archive.scope, "shards": shards,
                "native_rows": sum(s["rows"] for s in shards), "raw_rows_reconstructed": True,
                "synthetic_observations": 0, "known_hardware_fault_labels": 0,
                "missing_policy": "Absent rows stay absent; missing readings stay missing",
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(manifest_path, manifest)
    return manifest


def walk_cache(cache, manifest, *, full=False):
    tail, previous = None, None
    for entry in manifest["shards"]:
        p = Path(cache) / entry["file"]
        if sha256_file(p) != entry["sha256"]:
            raise ValueError("native cache no longer matches its verified manifest")
        frame = pd.read_parquet(p, columns=None if full else BASE_COLUMNS)
        group = entry["station_id"] + "|" + entry["source"]
        if group != previous:
            tail = None
        context = frame if tail is None else pd.concat([tail, frame], ignore_index=True)
        # 180 actual minutes suffice for max forecast/context lag 120 plus 30
        # residual-history minutes. This stores existing rows only, never a grid.
        tail = context.loc[context.timestamp >= context.timestamp.iloc[-1] - pd.Timedelta(minutes=180)].copy()
        previous = group
        yield entry, context.reset_index(drop=True), len(context) - len(frame)


def collect_samples(cache, manifest, cfg):
    chunks = {s: {h: defaultdict(list) for h in cfg.horizons} for s in ("train", "selection")}
    rng = np.random.default_rng(cfg.random_seed)
    for entry, frame, skip in walk_cache(cache, manifest):
        if entry["station_id"] in cfg.holdout_station_ids:
            continue
        splits = split_names(frame, cfg)
        for h in cfg.horizons:
            f = causal_features(frame, h, cfg)
            for split in chunks:
                indices = np.flatnonzero((np.arange(len(frame)) >= skip) & (splits == split) & f["complete"])
                if len(indices) > cfg.sample_per_month:
                    indices = np.sort(rng.choice(indices, cfg.sample_per_month, replace=False))
                if not len(indices):
                    continue
                target = frame[list(CHANNELS)].to_numpy(float, na_value=np.nan)
                qc = frame[[c + "__qc_accepted" for c in CHANNELS]].fillna(False).to_numpy(bool)
                for name, array in {"X": f["X"], "accepted": f["accepted"], "baseline": f["baseline"],
                                    "y": target, "target_accepted": qc}.items():
                    chunks[split][h][name].append(array[indices])
    return {s: {h: {k: np.concatenate(v) for k, v in values.items()} for h, values in horizons.items()}
            for s, horizons in chunks.items()}


def signal_walk(cache, manifest, models, cfg, *, full=False):
    state, prior_group = None, None
    for entry, frame, skip in walk_cache(cache, manifest, full=full):
        group = entry["station_id"] + "|" + entry["source"]
        if group != prior_group:
            state = None
        signals, qc, predictions, features = compute_signals(frame, models, cfg)
        current = frame.iloc[skip:].reset_index(drop=True)
        flats, state = flatline_durations(current, state)
        signals = {c: {s: v[skip:] for s, v in vals.items()} for c, vals in signals.items()}
        for c in CHANNELS:
            signals[c]["flatline_minutes"] = flats[c]
        qc = {c: {s: v[skip:] for s, v in vals.items()} for c, vals in qc.items()}
        predictions = {h: p[skip:] for h, p in predictions.items()}
        features = {h: {"accepted": f["accepted"][skip:], "baseline": f["baseline"][skip:],
                        "context_ids": {k: a[skip:] for k, a in f["context_ids"].items()}} for h, f in features.items()}
        prior_group = group
        yield entry, current, signals, qc, predictions, features


def calibrate(cache, manifest, models, cfg):
    def empty():
        return {c: {s: [] for s in SIGNALS} for c in CHANNELS}
    reference = defaultdict(empty)
    for entry, frame, signals, qc, _, _ in signal_walk(cache, manifest, models, cfg):
        if entry["station_id"] in cfg.holdout_station_ids:
            continue
        group = entry["station_id"] + "|" + entry["source"]
        mask = split_names(frame, cfg) == "calibration"
        if not mask.any():
            continue
        for c in CHANNELS:
            for s in SIGNALS:
                values = signals[c][s][mask & qc[c][s]]
                reference[group][c][s].append(values)
                reference["pooled_seen_groups"][c][s].append(values)
    return fit_thresholds(reference, cfg)


def long_alerts(frame):
    blocks = []
    for c in CHANNELS:
        block = frame[["group", "timestamp", "station_id", "source", "observation_id", "split"]].copy()
        block["channel"] = c
        block["is_alert"] = frame[c + "__alert"]
        block["score"] = frame[c + "__score"]
        block["reason_codes"] = frame[c + "__reason_codes"]
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def replay(cache, manifest, out, models, detector, cfg, *, fresh=False):
    from awsad.evaluation.real_events import extract_candidate_events, review_template, evaluate_reviewed_events
    target = out / "scored_observations"
    target.mkdir()
    rows, candidate_rows, channel_alerts, gaps, observed_minutes = 0, 0, 0, 0, 0
    all_events, shard_info, forecast = [], [], defaultdict(lambda: [0, 0., 0.])
    provider = defaultdict(lambda: {"accepted_alert": 0, "accepted_no_alert": 0,
                                  "rejected_alert": 0, "rejected_no_alert": 0, "unknown_or_unscored": 0})
    previous_timestamp = {}
    background_samples = []
    for entry, frame, signals, _, predictions, features in signal_walk(cache, manifest, models, cfg, full=True):
        scored = apply_thresholds(frame, signals, predictions, detector["thresholds"], cfg)
        if fresh:
            scored["split"] = "fresh_2025_replay"
        for h, f in features.items():
            for lag, ids in f["context_ids"].items():
                scored[f"prediction_{h}m__context_minus_{lag}m_observation_id"] = ids
        p = target / entry["file"]
        scored.to_parquet(p, index=False, compression="zstd")
        shard_info.append({"file": "scored_observations/" + p.name, "sha256": sha256_file(p),
                           "rows": len(scored), "group": scored.group.iloc[0], "start": entry["start"], "end": entry["end"]})
        long = long_alerts(scored)
        events = extract_candidate_events(long)
        all_events.append(events)
        rows += len(scored)
        candidate_rows += int(scored.is_candidate.sum())
        channel_alerts += sum(int(scored[c + "__alert"].sum()) for c in CHANNELS)
        group = scored.group.iloc[0]
        times = pd.DatetimeIndex(scored.timestamp)
        deltas = np.diff(times.as_unit("ns").asi8)
        gaps += int((deltas > 60_000_000_000).sum())
        if group in previous_timestamp and times[0] - previous_timestamp[group] > pd.Timedelta(minutes=1):
            gaps += 1
        previous_timestamp[group] = times[-1]
        observed_minutes += len(scored)
        for cidx, c in enumerate(CHANNELS):
            flagged = scored[c + "__alert"]
            accepted = scored[c + "__qc_accepted"].fillna(False)
            rejected = scored[c + "__qc_rejected"].fillna(False)
            counts = provider[group + ":" + c]
            for name, q in (("accepted", accepted), ("rejected", rejected)):
                counts[name + "_alert"] += int((q & flagged.fillna(False)).sum())
                counts[name + "_no_alert"] += int((q & ~flagged.fillna(True)).sum())
            counts["unknown_or_unscored"] += int((~(accepted | rejected) | flagged.isna()).sum())
            y = scored[c].to_numpy(float, na_value=np.nan)
            for h in cfg.horizons:
                ok = accepted.to_numpy() & features[h]["accepted"] & np.isfinite(predictions[h][:, cidx]) & np.isfinite(y)
                for split in scored.split.unique():
                    m = ok & scored.split.eq(split).to_numpy()
                    accumulator = forecast[f"{group}:{split}:{h}:{c}"]
                    accumulator[0] += int(m.sum())
                    accumulator[1] += float(np.abs(y[m] - predictions[h][m, cidx]).sum())
                    accumulator[2] += float(np.abs(y[m] - features[h]["baseline"][m, cidx]).sum())
        # Include a deterministic non-alert observation for review per shard;
        # its truth is unknown. This is a proposal, never a normal label.
        quiet = long.loc[long.is_alert.eq(False).fillna(False)]
        if not quiet.empty:
            sample = quiet.iloc[len(quiet) // 2]
            background_samples.append({"event_id": "unreviewed-" + entry["file"], "group": sample.group,
                "channel": sample.channel, "start": sample.timestamp, "end": sample.timestamp,
                "proposal_kind": "non_alert_observation", "first_observation_id": sample.observation_id,
                "last_observation_id": sample.observation_id})
        print(f"Scored {entry['file']}: {len(scored):,} real rows, {int(scored.is_candidate.sum()):,} candidate minutes", flush=True)
    events = pd.concat(all_events, ignore_index=True)
    # Month boundaries split exported event proposals conservatively. Never
    # merge across missing records, a split, or a station/source boundary.
    events.to_csv(out / "candidate_events.csv", index=False)
    proposals = pd.concat([events, pd.DataFrame(background_samples)], ignore_index=True)
    reviews = review_template(proposals)
    reviews.to_csv(out / "review_template.csv", index=False)
    empty_alerts = pd.DataFrame(columns=["group", "timestamp", "channel", "observation_id", "is_alert"])
    event_metrics = evaluate_reviewed_events(empty_alerts, reviews)
    metrics = {"dataset_policy": "real_observations_only", "status": "real_minute_replay_complete",
        "native_observations": rows, "candidate_minutes": candidate_rows,
        "candidate_fraction": candidate_rows / rows, "channel_alerts": channel_alerts,
        "candidate_event_proposals": len(events), "review_proposals": len(reviews),
        "observed_station_minutes": observed_minutes, "gaps_between_observed_records": gaps,
        "synthetic_observations": 0, "known_hardware_fault_labels": 0,
        "detector_config_sha256": config_fingerprint(cfg), "fresh_period": fresh,
        "real_event_evaluation": event_metrics,
        "provider_qc_agreement_counts": dict(provider),
        "provider_qc_semantics": "Quality evidence only; accepted does not mean fault-free; rejected does not mean hardware failure",
        "forecast_mae": {k: {"rows": v[0], "model": v[1] / v[0] if v[0] else None,
                              "persistence": v[2] / v[0] if v[0] else None} for k, v in forecast.items()},
        "limitations": ["Candidates are unconfirmed; fault precision/recall/F1 and false-alarm rate are not established",
                        "Empirical tail budget is not a guaranteed false-alarm rate",
                        "US SURFRAD sites, not validation on Indian IMD AWS hardware",
                        "Candidate proposals split conservatively at monthly artifact boundaries"]}
    write_json(out / "metrics.json", metrics)
    write_json(out / "provenance.json", {"native_manifest": manifest, "scored_shards": shard_info,
               "source_files": detector["source_files"], "attribution": "NOAA Global Monitoring Laboratory / SURFRAD; CC0 1.0"})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--replay-only", type=Path, help="frozen detector directory; requires a fresh start after calibration and 2024")
    args = parser.parse_args()
    blob = json.loads(args.config.read_text()) if args.config else {}
    for key in ("horizons", "context_minutes", "holdout_station_ids"):
        if key in blob:
            blob[key] = tuple(blob[key])
    cfg = MinuteConfig(**blob)
    cfg.validate()
    if args.replay_only:
        detector = json.loads((args.replay_only / "detector.json").read_text())
        if config_fingerprint(cfg) != detector["config_sha256"]:
            raise ValueError("frozen configuration mismatch")
        if not args.start or pd.Timestamp(args.start) < pd.Timestamp("2025-01-01T00:00:00Z"):
            raise ValueError("fresh replay must exclude previously examined 2024 test data")
        if sha256_file(args.replay_only / "models.joblib") != detector["models_sha256"]:
            raise ValueError("frozen model bytes changed")
        for name, digest in detector["source_files"].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError("source changed since freeze; review and refreeze before fresh replay")
    manifest = build_cache(args.archive_dir, args.cache_dir, start=args.start,
                           end=args.end or (None if args.replay_only else cfg.calibration_end))
    if args.cache_only:
        return
    if args.out is None:
        raise ValueError("--out is required")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("use a new artifact directory to preserve experiments")
    out.mkdir(parents=True)
    if args.replay_only:
        models = joblib.load(args.replay_only / "models.joblib")
        write_json(out / "detector.json", detector)
    else:
        if args.end and pd.Timestamp(args.end) > pd.Timestamp(cfg.calibration_end):
            raise ValueError("development replay cannot include previously examined test dates")
        samples = collect_samples(args.cache_dir, manifest, cfg)
        models, selection = fit_forecasters(samples, cfg)
        del samples
        thresholds = calibrate(args.cache_dir, manifest, models, cfg)
        joblib.dump(models, out / "models.joblib")
        files = ["src/awsad/minute_detection.py", "src/awsad/data/surfrad.py",
                 "src/awsad/data/surfrad_native.py", "src/awsad/evaluation/real_events.py", "scripts/run_minute_detection.py"]
        detector = {"dataset_policy": "real_observations_only", "config": asdict(cfg),
                    "config_sha256": config_fingerprint(cfg), "thresholds": thresholds, "model_selection": selection,
                    "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "models_sha256": sha256_file(out / "models.joblib"),
                    "source_files": {f: sha256_file(ROOT / f) for f in files},
                    "score_semantics": "Maximum signal/empirical-threshold ratio, not fault probability",
                    "calibration_semantics": "Provider-accepted original observations, not known-normal labels",
                    "forecast_features": "Only past original T/RH/station pressure and their differences",
                    "forecast_context_minutes": list(cfg.context_minutes), "test_used_for_selection": False}
        write_json(out / "detector.json", detector)
    metrics = replay(args.cache_dir, manifest, out, models, detector, cfg, fresh=bool(args.replay_only))
    print(json.dumps({k: metrics[k] for k in ("native_observations", "candidate_minutes", "candidate_event_proposals", "known_hardware_fault_labels")}, indent=2))


if __name__ == "__main__":
    main()
