"""Verify all cached feature rows, calibrate selected raw models, seal release run.

This final development step never accesses final-test sources. It rechecks every
original scenario copy against its stored hash and every feature row against the
current executable transform, including changes limited to JSON-null handling.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_sih_benchmark as runner
from awsad.benchmark.operational_model import OperationalPatternModel, causal_pattern_features
from awsad.benchmark.scenarios import CHANNELS
from awsad.benchmark.sih_evaluation import choose_threshold
from awsad.data.acquisition import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="artifacts_sih_tree_selected_20260930")
    parser.add_argument("--scenarios", default="data/sih_scenarios_v2_reviewed_20260930")
    parser.add_argument("--out", default="artifacts_sih_release_20260930")
    parser.add_argument("--verified-run", help="reuse exact feature proof only when source/data fingerprints still match")
    args = parser.parse_args()
    source, cache, output = Path(args.source_run), Path(args.scenarios), Path(args.out)
    if output.exists():
        raise FileExistsError("sealed outputs must use a new directory")
    original = json.loads((source / "frozen.json").read_text())
    if sha256_file(source / "pattern_model.joblib") != original["model_sha256"]:
        raise ValueError("selected model changed")
    if original["environment"]["packages"] != runner.runtime_versions():
        raise ValueError("selected model dependency versions changed")
    base = OperationalPatternModel.load(source / "pattern_model.joblib")
    previous = json.loads((Path(args.verified_run) / "frozen.json").read_text()) if args.verified_run else None
    if previous is not None and (
            previous.get("parent_selected_run_sha256") != sha256_file(source / "frozen.json")
            or previous.get("source_fingerprints", {}).get("src/awsad/benchmark/operational_model.py")
                != sha256_file(ROOT / "src/awsad/benchmark/operational_model.py")
            or previous.get("scenario_metadata_sha256") != original["scenario_metadata_sha256"]
            or previous.get("feature_reverification") != "exact float32 equality including NaN at every stored development row"):
        raise ValueError("prior feature proof does not match current transform/selected model/data")
    partitions, verified_rows = {}, 0
    for split, expected in original["scenario_metadata_sha256"].items():
        metadata_path = cache / f"{split}.json"
        if sha256_file(metadata_path) != expected:
            raise ValueError("selected model's development data changed")
        metadata = json.loads(metadata_path.read_text())
        runner.verify_partition_artifacts(cache, metadata)
        data = dict(np.load(cache / f"{split}.npz", allow_pickle=False))
        runner.validate_cached_arrays(data, metadata)
        for index, event in enumerate(metadata["events"] if previous is None else []):
            scenario = pd.read_parquet(cache / event["artifact"])
            values = scenario[["timestamp", *["scenario__" + channel for channel in CHANNELS]]].rename(
                columns={"scenario__" + channel: channel for channel in CHANNELS})
            positions = data["positions"][data["scenario_indices"] == index]
            expected_features = data["X"][data["scenario_indices"] == index]
            actual = causal_pattern_features(values).iloc[positions].to_numpy(np.float32)
            if not np.array_equal(expected_features, actual, equal_nan=True):
                raise ValueError(f"feature transform differs from selected training data: {event['scenario_id']}")
            verified_rows += len(actual)
        if previous is not None:
            verified_rows += len(data["y"])
        partitions[split] = data, metadata
        print(f"Reverified current feature transform: {split}, {len(data['y'])} rows", flush=True)
    runner.audit_partitions(partitions)
    if previous is not None and verified_rows != previous.get("verified_feature_rows"):
        raise ValueError("previous feature proof row count differs")
    calibration, calibration_meta = partitions["calibration"]
    model = OperationalPatternModel(base.model, base.features, metadata=base.metadata)
    with threadpool_limits(limits=4):
        model.calibrate(runner.feature_frame(calibration, calibration_meta), calibration["y"])
        scores = model.predict_features(runner.feature_frame(calibration, calibration_meta))["scenario_score"]
        model.threshold, operating_point = choose_threshold(calibration["y"] != "no_injection", scores)
        selection_report = runner.score_dataset(model, *partitions["selection"])
        calibration_report = runner.score_dataset(model, *partitions["calibration"])
    model.metadata["calibration_method"] = "monotone binary logistic + conditional type temperature"
    output.mkdir()
    model.save(output / "pattern_model.joblib")
    if sha256_file(source / "isolation_baseline.joblib") != original["isolation_baseline_sha256"]:
        raise ValueError("selected baseline changed")
    (output / "isolation_baseline.joblib").write_bytes((source / "isolation_baseline.joblib").read_bytes())
    freeze = {**original, "source_fingerprints": runner.source_fingerprints(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_selected_run_sha256": sha256_file(source / "frozen.json"),
        "model_sha256": sha256_file(output / "pattern_model.joblib"),
        "temperature": model.temperature, "binary_calibration": model.binary_calibration,
        "calibration_method": model.metadata["calibration_method"],
        "decision_threshold": model.threshold, "calibration_threshold_metrics": operating_point,
        "verified_feature_rows": verified_rows, "feature_reverification": "exact float32 equality including NaN at every stored development row",
        "seal_script_sha256": sha256_file(Path(__file__)), "final_test_status_at_freeze": "unexamined"}
    if previous is not None:
        freeze["reused_feature_proof_sha256"] = sha256_file(Path(args.verified_run) / "frozen.json")
    runner.write(output / "frozen.json", freeze)
    runner.write(output / "selection.json", selection_report)
    runner.write(output / "calibration.json", calibration_report)
    runner.write(output / "metrics.json", {"schema_version": "sih_benchmark_v2", "policy": freeze["policy"],
        "status": "frozen_awaiting_final_test", "split_audit": freeze["split_audit"],
        "selection": selection_report, "calibration": calibration_report, "final_test": None,
        "disclaimer": "Synthetic software-pattern evidence only; real hardware-fault accuracy remains unknown"})
    print(f"Sealed {output}; final data remains unexamined.", flush=True)


if __name__ == "__main__":
    main()
