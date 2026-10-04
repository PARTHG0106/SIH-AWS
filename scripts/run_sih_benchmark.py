"""Build, select and independently evaluate the SIH scenario model.

Develop never consumes final-test sources. Evaluate requires a frozen model and
configuration hash and writes a one-time result to prevent silent test retuning.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version as package_version
import json
import os
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from awsad.benchmark.scenarios import (FAULTS, HISTORY, VERSION, SourceWindow,
    WINDOW, generate, partition as source_partition, source_windows, stable_seed)
from awsad.benchmark.operational_model import (FEATURE_VERSION, OperationalPatternModel,
    causal_pattern_features, fit_temperature)
from awsad.benchmark.sih_evaluation import (choose_threshold, evaluate, point_metrics,
    simple_scores)
from awsad.data.acquisition import sha256_file
from awsad.benchmark.injection_eval import load_detector, _score
from awsad.benchmark.scenarios import CHANNELS

FINAL_TEST_SCOPE = {"source_product": "noaa_surfrad_native",
    "stations": ["bon", "fpk", "gwn"],
    "start": "2025-02-01T00:00:00Z", "end_exclusive": "2025-03-01T00:00:00Z"}
TEST_REGISTRY = ROOT / "artifacts_sih_test_registry"


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_fingerprints():
    # Conservatively seal the entire local package, including the existing
    # minute detector and its imports. Sealing only model artifacts misses
    # changes in the executable baseline scoring implementation.
    paths = sorted((ROOT / "src" / "awsad").rglob("*.py"))
    paths.append(ROOT / "scripts" / "run_sih_benchmark.py")
    paths.append(ROOT / "scripts" / "run_minute_detection.py")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def runtime_versions():
    return {name: package_version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "pyarrow")}


def verify_partition_artifacts(destination, metadata, *, copies=True):
    root = Path(destination).resolve()
    split = metadata["spec"]["split"]
    arrays_path = root / f"{split}.npz"
    if sha256_file(arrays_path) != metadata["arrays_sha256"]:
        raise ValueError(f"{split} cached arrays changed")
    if copies:
        for event in metadata["events"]:
            path = (root / event["artifact"]).resolve()
            if path.parent != root / split or sha256_file(path) != event["artifact_sha256"]:
                raise ValueError(f"{split} scenario copy hash/path mismatch: {event['artifact']}")


def validate_cached_arrays(dataset, metadata):
    n = metadata["rows"]
    if set(dataset) != {"X", "y", "scenario_indices", "positions"} or dataset["X"].shape != (n, len(metadata["features"])):
        raise ValueError("cached scenario feature shape/schema mismatch")
    if any(dataset[key].shape != (n,) for key in ("y", "scenario_indices", "positions")):
        raise ValueError("cached scenario labels/positions are misaligned")
    indices, positions = dataset["scenario_indices"], dataset["positions"]
    if (not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(positions.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(metadata["events"]))
            or np.any(positions < HISTORY) or np.any(positions >= WINDOW)
            or not np.isin(dataset["y"], ("no_injection", *FAULTS)).all()):
        raise ValueError("cached scenario labels/indices are invalid")
    for index in np.unique(indices):
        if not np.all(np.diff(positions[indices == index]) > 0):
            raise ValueError("cached scenario positions must be unique and chronological")


def _scope_overlap(left, right):
    if left["source_product"] != right["source_product"] or not set(left["stations"]) & set(right["stations"]):
        return False
    return max(pd.Timestamp(left["start"]), pd.Timestamp(right["start"])) < min(
        pd.Timestamp(left["end_exclusive"]), pd.Timestamp(right["end_exclusive"]))


@contextmanager
def _registry_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".registry.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("holdout registry is locked; no new final data may be read") from error
    try:
        os.close(handle)
        yield
    finally:
        lock.unlink()


def _atomic_registry_write(path, receipt):
    write(path, receipt)


def claim_final_test(scope, frozen_sha256, acquisition_manifest_sha256, output, *, registry=None):
    """Record consumption before parsing any final measurement or scenario.

    A failed/interrupted attempt may resume only with the exact same frozen
    model/protocol AND raw-acquisition-manifest bytes. Changing output directories,
    injection seeds or station subsets cannot create a new untouched holdout.
    """
    directory = Path(registry) if registry is not None else TEST_REGISTRY
    scope = {**scope, "stations": sorted(set(scope["stations"]))}
    if not scope["stations"] or pd.Timestamp(scope["start"]) >= pd.Timestamp(scope["end_exclusive"]):
        raise ValueError("empty or invalid final station/time scope")
    key = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    path = directory / f"{key}.json"
    with _registry_lock(directory):
        receipt = None
        for prior_path in directory.glob("*.json"):
            prior = json.loads(prior_path.read_text())
            if not _scope_overlap(scope, prior["scope"]):
                continue
            if prior_path == path and prior["status"] == "started":
                raise RuntimeError("final attempt is still marked started; do not run concurrent evaluations or erase an interrupted receipt")
            if (prior_path != path or prior["status"] == "completed"
                    or prior["frozen_sha256"] != frozen_sha256
                    or prior["test_acquisition_manifest_sha256"] != acquisition_manifest_sha256
                    or prior["status"] != "failed"):
                raise FileExistsError("final source station/time scope already consumed; reserve a new untouched holdout")
            receipt = prior
        if receipt is None:
            receipt = {"schema_version": "sih_holdout_consumption_v2", "scope": scope,
                       "frozen_sha256": frozen_sha256, "test_acquisition_manifest_sha256": acquisition_manifest_sha256,
                       "attempts": [], "test_consumed_do_not_retune": True}
        receipt["status"] = "started"
        receipt["attempts"].append({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "output_directory": str(Path(output).resolve())})
        _atomic_registry_write(path, receipt)
    return path


def build_native_test_cache(archive, cache, scope):
    """Called only inside the already-claimed final evaluation transaction.

    build_cache validates source evidence and its linkage even on cache reuse;
    native daily values are parsed only if a cache must be constructed.
    """
    helper = ROOT / "scripts" / "run_minute_detection.py"
    spec = importlib.util.spec_from_file_location("sih_native_cache_builder", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_cache(archive, cache, start=scope["start"], end=scope["end_exclusive"])


def bind_native_test_cache(receipt_path, frozen_hash, acquisition_hash, cache, manifest, scope):
    """Seal the constructed cache separately from the raw acquisition receipt."""
    evidence, identity = manifest.get("evidence", {}), manifest.get("identity", {})
    if (evidence.get("manifest_path") != "surfrad_acquisition.json"
            or evidence.get("manifest_sha256") != acquisition_hash
            or not evidence.get("index_verified")
            or not identity.get("source_fingerprint")
            or identity["source_fingerprint"] != evidence.get("evidence_fingerprint")
            or identity.get("start") != scope["start"] or identity.get("end") != scope["end_exclusive"]
            or sorted(manifest.get("scope", {}).get("stations", [])) != sorted(scope["stations"])
            or not manifest.get("raw_rows_reconstructed")
            or manifest.get("synthetic_observations") != 0):
        raise ValueError("native test cache does not match the claimed raw acquisition and station/time scope")
    native_path = Path(cache) / "native_manifest.json"
    if json.loads(native_path.read_text()) != manifest:
        raise ValueError("native test manifest changed during cache construction")
    native_hash = sha256_file(native_path)
    with _registry_lock(receipt_path.parent):
        receipt = json.loads(receipt_path.read_text())
        if (receipt["status"] != "started" or receipt["frozen_sha256"] != frozen_hash
                or receipt["test_acquisition_manifest_sha256"] != acquisition_hash):
            raise ValueError("holdout claim changed before native cache binding")
        if receipt.get("native_cache_manifest_sha256", native_hash) != native_hash:
            raise ValueError("native test cache changed after an earlier attempt consumed it")
        receipt["native_cache_manifest_sha256"] = native_hash
        receipt["native_cache_evidence_fingerprint"] = identity["source_fingerprint"]
        _atomic_registry_write(receipt_path, receipt)
    return native_hash


def finish_final_test(path, frozen_sha256, status, *, final_path=None, error=None):
    if status not in {"completed", "failed"}:
        raise ValueError("invalid holdout completion status")
    with _registry_lock(path.parent):
        receipt = json.loads(path.read_text())
        if receipt["frozen_sha256"] != frozen_sha256 or receipt["status"] != "started":
            raise ValueError("holdout claim changed during evaluation")
        receipt["status"] = status
        receipt["attempts"][-1].update(status=status, ended_at_utc=datetime.now(timezone.utc).isoformat())
        if final_path is not None:
            receipt["final_result_sha256"] = sha256_file(final_path)
        if error is not None:
            receipt["attempts"][-1]["error"] = str(error)
        _atomic_registry_write(path, receipt)


def make_partition(cache, destination, split, windows_per_shard, seed):
    destination.mkdir(parents=True, exist_ok=True)
    spec = {"cache_schema_version": 2, "version": VERSION, "feature_version": FEATURE_VERSION, "split": split,
            "windows_per_shard": windows_per_shard, "seed": seed,
            "source_manifest_sha256": sha256_file(Path(cache) / "native_manifest.json"),
            "generator_sha256": sha256_file(ROOT / "src/awsad/benchmark/scenarios.py"),
            "features_sha256": sha256_file(ROOT / "src/awsad/benchmark/operational_model.py")}
    meta_path, arrays_path = destination / f"{split}.json", destination / f"{split}.npz"
    if meta_path.exists() and arrays_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata["spec"] != spec:
            raise ValueError(f"existing {split} scenario cache differs; use a new scenario directory")
        verify_partition_artifacts(destination, metadata)
        with np.load(arrays_path, allow_pickle=False) as archive:
            dataset = dict(archive)
        validate_cached_arrays(dataset, metadata)
        return dataset, metadata
    arrays, labels, scenario_idx, row_positions = [], [], [], []
    events, exclusions, windows, feature_names = [], [], [], None
    scenario_dir = destination / split
    scenario_dir.mkdir(exist_ok=True)
    for source in source_windows(cache, {split}, windows_per_shard=windows_per_shard, seed=seed):
        if isinstance(source, dict):
            exclusions.append(source)
            continue
        if source.frame[["station_id", "source"]].drop_duplicates().shape[0] != 1:
            raise ValueError("one original station/source is required per source window")
        station = str(source.frame.station_id.iloc[0])
        if any(source_partition(station, stamp) != split for stamp in
               (source.frame.timestamp.iloc[0], source.frame.timestamp.iloc[-1])):
            raise ValueError("source observations cross or disagree with the declared partition")
        windows.append({"window_id": source.window_id, "source_file": source.source_file,
                        "station_id": station, "source": str(source.frame.source.iloc[0]),
                        "source_sha256": source.source_sha256,
                        "start": source.frame.timestamp.iloc[0].isoformat(),
                        "end": source.frame.timestamp.iloc[-1].isoformat(),
                        "observation_ids_sha256": hashlib.sha256("\n".join(source.frame.observation_id).encode()).hexdigest()})
        for fault in ("no_injection", *FAULTS):
            scenario = generate(source, fault, seed=seed)
            features = causal_pattern_features(scenario.values)
            feature_names = list(features)
            valid = np.arange(len(features)) >= HISTORY
            # Subsample fit rows only, retaining every rare spike and intervention
            # boundary. Selection/calibration/test always keep every scored row.
            if split == "fit":
                valid &= ((np.arange(len(features)) % 3 == 0) | (scenario.labels == "spike"))
            index = len(events)
            arrays.append(features.loc[valid].to_numpy(np.float32))
            labels.append(scenario.labels[valid].astype("U16"))
            scenario_idx.append(np.full(valid.sum(), index, np.int32))
            row_positions.append(np.flatnonzero(valid).astype(np.int16))
            filename = f"{index:05d}_{fault}.parquet"
            scenario.export().to_parquet(scenario_dir / filename, index=False)
            events.append({**scenario.event, "artifact": f"{split}/{filename}",
                           "artifact_sha256": sha256_file(scenario_dir / filename)})
        print(f"{split}: {len(windows)} source windows / {len(events)} scenario copies", flush=True)
    if not arrays:
        raise ValueError(f"no eligible source windows for {split}")
    dataset = {"X": np.vstack(arrays), "y": np.concatenate(labels),
               "scenario_indices": np.concatenate(scenario_idx), "positions": np.concatenate(row_positions)}
    np.savez_compressed(arrays_path, **dataset)
    metadata = {"spec": spec, "features": feature_names, "events": events,
                "source_windows": windows, "excluded_windows": exclusions,
                "rows": len(dataset["y"]), "arrays_sha256": sha256_file(arrays_path),
                "semantics": "scenario targets only; all original baseline columns remain unchanged"}
    write(meta_path, metadata)
    return dataset, metadata


def feature_frame(dataset, metadata):
    return pd.DataFrame(dataset["X"], columns=metadata["features"])


def score_dataset(model, dataset, metadata):
    prediction = model.predict_features(feature_frame(dataset, metadata))
    return evaluate(dataset["y"], prediction["is_candidate"], prediction["scenario_score"],
        prediction["pattern"], dataset["scenario_indices"], dataset["positions"], metadata["events"])


def audit_partitions(partitions):
    seen, intervals = set(), {}
    for split, (_, metadata) in partitions.items():
        for window in metadata["source_windows"]:
            if window["window_id"] in seen:
                raise ValueError("source window leaked across partitions")
            seen.add(window["window_id"])
            station = window.get("station_id", window["window_id"].split("|", 1)[0])
            start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
            if start.tzinfo is None or end.tzinfo is None or start > end:
                raise ValueError("invalid source-window interval")
            intervals.setdefault(station, []).append((start, end, split, window["window_id"]))
    for windows in intervals.values():
        ordered = sorted(windows)
        for before, after in zip(ordered, ordered[1:]):
            if after[0] <= before[1]:
                raise ValueError(f"overlapping source histories: {before[3]} and {after[3]}")
    return {"unique_source_windows": len(seen), "overlapping_source_windows": 0,
            "split_before_generation": True,
            "context_policy": "interval-checked disjoint source windows including all warm-up history; first 180 minutes excluded from scored targets"}


def baseline_threshold(truth, score):
    grid = np.unique(np.r_[np.quantile(score, np.linspace(0, 1, 301)), score.max() + 1.])
    choices = [(point_metrics(truth, score >= threshold), float(threshold)) for threshold in grid]
    allowed = [(metric, threshold) for metric, threshold in choices
               if metric["unmodified_row_candidate_rate"] <= .02]
    return max(allowed, key=lambda item: item[0]["f1"])[1]


def frozen_scores(scenario_root, dataset, metadata, artifact_dir):
    cfg, models, thresholds = load_detector(artifact_dir)
    scores, flags = np.zeros(len(dataset["y"])), np.zeros(len(dataset["y"]), bool)
    for index, event in enumerate(metadata["events"]):
        path = Path(scenario_root) / event["artifact"]
        if sha256_file(path) != event["artifact_sha256"]:
            raise ValueError("scenario copy hash mismatch")
        copy = pd.read_parquet(path)
        for channel in CHANNELS:
            copy[channel] = copy["scenario__" + channel]
        scored = _score(copy, models, thresholds, cfg)
        selected = dataset["scenario_indices"] == index
        positions = dataset["positions"][selected]
        scores[selected] = np.nan_to_num(scored.anomaly_score.to_numpy(float)[positions], nan=0.)
        flags[selected] = scored.is_candidate.to_numpy(bool)[positions]
    return scores, flags


def develop(args):
    output = Path(args.out)
    if (output / "frozen.json").exists():
        raise FileExistsError("frozen runs are immutable; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    partitions = {split: make_partition(args.cache, Path(args.scenarios), split,
                  args.windows_per_shard, args.seed) for split in ("fit", "selection", "calibration")}
    split_audit = audit_partitions(partitions)
    train, train_meta = partitions["fit"]
    selection, selection_meta = partitions["selection"]
    calibration, calibration_meta = partitions["calibration"]
    features = train_meta["features"]
    labels, counts = np.unique(train["y"], return_counts=True)
    # Square-root class weights help rare impulses without forcing equal priors
    # at deployment. Temperature is calibrated on untouched calibration rows.
    weights = {label: np.sqrt(len(train["y"]) / count) for label, count in zip(labels, counts)}
    sample_weight = np.array([weights[label] for label in train["y"]])
    trials, best = [], None
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        for leaves in (15, 31):
            clf = HistGradientBoostingClassifier(max_iter=130, max_leaf_nodes=leaves,
                learning_rate=.08, min_samples_leaf=20, l2_regularization=2.,
                early_stopping=False, random_state=args.seed)
            clf.fit(train["X"], train["y"], sample_weight=sample_weight)
            predicted = clf.predict(selection["X"])
            macro = float(f1_score(selection["y"], predicted, labels=list(FAULTS), average="macro", zero_division=0))
            trial = {"max_leaf_nodes": leaves, "selection_macro_f1_faults": macro}
            trials.append(trial)
            print("model selection:", trial, flush=True)
            if best is None or macro > best[0]:
                best = (macro, clf, leaves)
        model = OperationalPatternModel(best[1], features)
        raw = model.model.predict_proba(calibration["X"])
        class_index = {label: index for index, label in enumerate(model.classes)}
        target_indices = np.array([class_index[label] for label in calibration["y"]])
        model.temperature = fit_temperature(raw, target_indices)
        probabilities = model.predict_features(feature_frame(calibration, calibration_meta))["scenario_score"]
        model.threshold, threshold_metrics = choose_threshold(calibration["y"] != "no_injection", probabilities)
        model.metadata = {"policy": "synthetic_scenarios_on_original_observations", "feature_version": FEATURE_VERSION,
                          "selection": trials, "station_holdout": "gwn", "scenario_version": VERSION}
        model.save(output / "pattern_model.joblib")
        selection_report = score_dataset(model, selection, selection_meta)
        calibration_report = score_dataset(model, calibration, calibration_meta)
    baseline_thresholds = {}
    cal_features = feature_frame(calibration, calibration_meta)
    for name, score in simple_scores(cal_features).items():
        truth = calibration["y"] != "no_injection"
        baseline_thresholds[name] = baseline_threshold(truth, score)
    baseline_features = [name for name in features if any(token in name for token in (":z_", ":delta_", ":missing", ":repeat_", ":short_long_", ":range_excess"))]
    background_indices = {index for index, event in enumerate(train_meta["events"]) if event["fault"] == "no_injection"}
    background = np.isin(train["scenario_indices"], list(background_indices))
    reference = feature_frame(train, train_meta).loc[background, baseline_features]
    isolation = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              RobustScaler(), IsolationForest(n_estimators=150, max_samples=512,
                              random_state=args.seed, n_jobs=4))
    isolation.fit(reference)
    isolation_scores = -isolation.score_samples(cal_features[baseline_features])
    baseline_thresholds["isolation_forest"] = baseline_threshold(calibration["y"] != "no_injection", isolation_scores)
    joblib.dump({"model": isolation, "features": baseline_features}, output / "isolation_baseline.joblib")
    training_seconds = time.perf_counter() - started
    freeze = {"schema_version": "sih_benchmark_v2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "policy": "synthetic_scenarios_on_original_observations", "seed": args.seed,
              "windows_per_shard": args.windows_per_shard, "final_windows_per_shard": args.test_windows_per_shard,
              "split_audit": split_audit, "source_fingerprints": source_fingerprints(),
              "model_sha256": sha256_file(output / "pattern_model.joblib"),
              "isolation_baseline_sha256": sha256_file(output / "isolation_baseline.joblib"),
              "frozen_minute_baseline": {name: sha256_file(Path(args.minute_artifacts) / name)
                                         for name in ("models.joblib", "detector.json", "config.json")},
              "scenario_metadata_sha256": {split: sha256_file(Path(args.scenarios) / f"{split}.json") for split in partitions},
              "baseline_thresholds": baseline_thresholds, "temperature": model.temperature,
              "decision_threshold": model.threshold, "calibration_threshold_metrics": threshold_metrics,
              "selection_trials": trials, "training_seconds": training_seconds,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "packages": runtime_versions()},
              "final_test_period": {"start": "2025-02-01T00:00:00Z", "end": "2025-03-01T00:00:00Z"},
              "final_test_scope": FINAL_TEST_SCOPE,
              "final_test_status_at_freeze": "unexamined",
              "consumption_registry": str(TEST_REGISTRY.relative_to(ROOT))}
    write(output / "selection.json", selection_report)
    write(output / "calibration.json", calibration_report)
    write(output / "frozen.json", freeze)
    write(output / "metrics.json", {"schema_version": "sih_benchmark_v2", "policy": freeze["policy"],
        "status": "frozen_awaiting_final_test", "split_audit": split_audit, "selection": selection_report,
        "calibration": calibration_report, "final_test": None,
        "disclaimer": "Synthetic software-pattern evidence only; real hardware-fault accuracy remains unknown"})
    print(f"Frozen model written: {output}. Final test has not been consumed.", flush=True)


def evaluate_final(args):
    output = Path(args.out)
    freeze = json.loads((output / "frozen.json").read_text())
    final_path = output / "final_test.json"
    if final_path.exists():
        raise FileExistsError("final test already examined; preserve this result and reserve a new test for later tuning")
    if freeze["source_fingerprints"] != source_fingerprints() or freeze["model_sha256"] != sha256_file(output / "pattern_model.joblib"):
        raise ValueError("model or protocol source changed since freeze")
    if freeze["environment"].get("packages") != runtime_versions() or freeze["environment"]["python"] != sys.version:
        raise ValueError("Python or scoring dependencies changed since freeze")
    for field, filename in (("selection_script_sha256", "select_sih_model.py"),
                            ("seal_script_sha256", "seal_sih_model.py")):
        if field in freeze and sha256_file(ROOT / "scripts" / filename) != freeze[field]:
            raise ValueError("selection/calibration sealing script changed since freeze")
    if freeze.get("final_test_scope") != FINAL_TEST_SCOPE:
        raise ValueError("final station/time scope differs from the frozen protocol")
    model = OperationalPatternModel.load(output / "pattern_model.joblib")
    if sha256_file(output / "isolation_baseline.joblib") != freeze["isolation_baseline_sha256"]:
        raise ValueError("isolation baseline changed after freeze")
    isolation = joblib.load(output / "isolation_baseline.joblib")
    for name, expected in freeze["frozen_minute_baseline"].items():
        if sha256_file(Path(args.minute_artifacts) / name) != expected:
            raise ValueError("frozen minute baseline changed after freeze")
    development = {}
    for split, expected in freeze["scenario_metadata_sha256"].items():
        metadata_path = Path(args.scenarios) / f"{split}.json"
        if sha256_file(metadata_path) != expected:
            raise ValueError("development scenario metadata changed after freeze")
        metadata = json.loads(metadata_path.read_text())
        verify_partition_artifacts(args.scenarios, metadata, copies=False)
        development[split] = (None, metadata)
    if set(development) != {"fit", "selection", "calibration"}:
        raise ValueError("all three frozen development partitions are required")
    # The downloaded acquisition inventory exists before any native cache. Its
    # byte hash is metadata-only; claim consumption before even build_cache can
    # parse an original observation, then bind the resulting native manifest.
    frozen_hash = sha256_file(output / "frozen.json")
    acquisition_path = Path(args.test_archive) / "surfrad_acquisition.json"
    acquisition_hash = sha256_file(acquisition_path)
    receipt_path = claim_final_test(freeze["final_test_scope"], frozen_hash,
                                    acquisition_hash, output)
    try:
        scope = freeze["final_test_scope"]
        native_manifest = build_native_test_cache(args.test_archive, args.test_cache, scope)
        if sha256_file(acquisition_path) != acquisition_hash:
            raise ValueError("raw test acquisition manifest changed after the holdout claim")
        test_manifest_hash = bind_native_test_cache(receipt_path, frozen_hash,
            acquisition_hash, args.test_cache, native_manifest, scope)
        partitions = {split: make_partition(args.test_cache, Path(args.scenarios), split,
            freeze["final_windows_per_shard"], freeze["seed"]) for split in ("test_temporal", "test_station")}
        split_audit = audit_partitions({**development, **partitions})
        results, integrity = {}, {}
        for split, (_, metadata) in partitions.items():
            if metadata["spec"]["source_manifest_sha256"] != test_manifest_hash:
                raise ValueError("test inventory changed after the holdout claim")
            for window in metadata["source_windows"]:
                if (window["station_id"] not in scope["stations"]
                        or pd.Timestamp(window["start"]) < pd.Timestamp(scope["start"])
                        or pd.Timestamp(window["end"]) >= pd.Timestamp(scope["end_exclusive"])):
                    raise ValueError("test source window lies outside the frozen station/time scope")
            integrity[split] = {"metadata_sha256": sha256_file(Path(args.scenarios) / f"{split}.json"),
                                "arrays_sha256": metadata["arrays_sha256"],
                                "source_manifest_sha256": metadata["spec"]["source_manifest_sha256"]}
        with threadpool_limits(limits=4):
            for split, (dataset, metadata) in partitions.items():
                report = score_dataset(model, dataset, metadata)
                baselines = {}
                frame = feature_frame(dataset, metadata)
                method_scores = simple_scores(frame)
                method_scores["isolation_forest"] = -isolation["model"].score_samples(frame[isolation["features"]])
                for name, scores in method_scores.items():
                    threshold = freeze["baseline_thresholds"][name]
                    baselines[name] = evaluate(dataset["y"], scores >= threshold, scores, None,
                        dataset["scenario_indices"], dataset["positions"], metadata["events"])
                    # Raw z/step magnitudes are not probabilities.
                    baselines[name].pop("reliability", None)
                scores, flags = frozen_scores(args.scenarios, dataset, metadata, args.minute_artifacts)
                baselines["frozen_minute_detector"] = evaluate(dataset["y"], flags, scores, None,
                    dataset["scenario_indices"], dataset["positions"], metadata["events"])
                baselines["frozen_minute_detector"].pop("reliability", None)
                learned = model.predict_features(frame)
                combined = flags | learned["is_candidate"]
                combined_score = np.maximum(scores, learned["scenario_score"] / max(model.threshold, 1e-12))
                baselines["combined_inspection_policy"] = evaluate(dataset["y"], combined,
                    combined_score, None, dataset["scenario_indices"], dataset["positions"], metadata["events"])
                baselines["combined_inspection_policy"].pop("reliability", None)
                baselines["combined_inspection_policy"]["semantics"] = (
                    "Predeclared union of separately labelled frozen-detector and learned-pattern review proposals; score is an exceedance ratio, not probability")
                results[split] = {"model": report, "baselines": baselines,
                                  "source_windows": len(metadata["source_windows"]), "rows": metadata["rows"]}
        final = {"status": "final_test_complete", "frozen_sha256": frozen_hash,
                 "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "partitions": results,
                 "partition_integrity": integrity, "split_audit": split_audit,
                 "test_source_manifest_sha256": test_manifest_hash,
                 "test_acquisition_manifest_sha256": acquisition_hash,
                 "native_cache_evidence_fingerprint": native_manifest["identity"]["source_fingerprint"], "scope": scope,
                 "consumption_receipt": str(receipt_path.relative_to(ROOT)),
                 "policy": freeze["policy"], "test_consumed_do_not_retune": True}
        write(final_path, final)
    except BaseException as error:
        finish_final_test(receipt_path, frozen_hash, "failed", error=error)
        raise
    finish_final_test(receipt_path, frozen_hash, "completed", final_path=final_path)
    metrics = json.loads((output / "metrics.json").read_text())
    metrics.update(status="final_test_complete", final_test=final)
    write(output / "metrics.json", metrics)
    print(json.dumps({key: value["model"]["point"] for key, value in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("develop", "evaluate"), required=True)
    parser.add_argument("--cache", default="data/native_minute_20260928")
    parser.add_argument("--minute-artifacts", default="artifacts_minute_20260928")
    parser.add_argument("--test-archive", default="data/raw/surfrad_sih_test_202502")
    parser.add_argument("--test-cache", default="data/native_minute_sih_test_202502")
    parser.add_argument("--scenarios", default="data/sih_scenarios_v2_20260930")
    parser.add_argument("--out", default="artifacts_sih_20260930")
    parser.add_argument("--windows-per-shard", type=int, default=4)
    parser.add_argument("--test-windows-per-shard", type=int, default=12)
    parser.add_argument("--seed", type=int, default=26073)
    args = parser.parse_args()
    if args.windows_per_shard < 1 or args.test_windows_per_shard < 1:
        parser.error("window counts must be positive")
    (develop if args.stage == "develop" else evaluate_final)(args)


if __name__ == "__main__":
    main()
