"""Compare the fitted multiclass model with independent detection/type heads.

Selection uses development validation only. Existing probability calibration is
discarded. The selected raw model is calibrated once on the separate calibration
partition, then sealed for final evaluation. No final source is accessed here.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from threadpoolctl import threadpool_limits

import run_sih_benchmark as runner
from awsad.benchmark.operational_model import OperationalPatternModel, fit_temperature
from awsad.benchmark.pattern_heads import TwoHeadPatternClassifier
from awsad.benchmark.scenarios import FAULTS
from awsad.benchmark.sih_evaluation import choose_threshold
from awsad.data.acquisition import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="artifacts_sih_20260930")
    parser.add_argument("--scenarios", default="data/sih_scenarios_v2_reviewed_20260930")
    parser.add_argument("--out", default="artifacts_sih_selected_20260930")
    parser.add_argument("--family", choices=("two_head", "trees"), default="trees")
    args = parser.parse_args()
    source, output, cache = Path(args.source_run), Path(args.out), Path(args.scenarios)
    if output.exists():
        raise FileExistsError("use a new selected-run output directory")
    original = json.loads((source / "frozen.json").read_text())
    if sha256_file(source / "pattern_model.joblib") != original["model_sha256"]:
        raise ValueError("base fitted model differs from its source run")
    partitions = {}
    for split, expected in original["scenario_metadata_sha256"].items():
        if sha256_file(cache / f"{split}.json") != expected:
            raise ValueError("development data differs from source fitted model")
        metadata = json.loads((cache / f"{split}.json").read_text())
        runner.verify_partition_artifacts(cache, metadata)
        data = dict(np.load(cache / f"{split}.npz", allow_pickle=False))
        runner.validate_cached_arrays(data, metadata)
        partitions[split] = data, metadata
    runner.audit_partitions(partitions)
    fit, fit_meta = partitions["fit"]
    selection, selection_meta = partitions["selection"]
    calibration, calibration_meta = partitions["calibration"]
    base = OperationalPatternModel.load(source / "pattern_model.joblib")
    if base.features != fit_meta["features"]:
        raise ValueError("feature schema differs from fitted source model")
    labels, counts = np.unique(fit["y"], return_counts=True)
    weights = dict(zip(labels, np.sqrt(len(fit["y"]) / counts)))
    sample_weight = np.array([weights[value] for value in fit["y"]])
    trials = []
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        if args.family == "two_head":
            alternatives = [("separate_detection_and_type_heads", TwoHeadPatternClassifier(seed=original["seed"]))]
        else:
            alternatives = [
                ("extra_trees", ExtraTreesClassifier(n_estimators=200, max_depth=24,
                    min_samples_leaf=5, max_features=.8, n_jobs=4, random_state=original["seed"])),
                ("random_forest", RandomForestClassifier(n_estimators=160, max_depth=24,
                    min_samples_leaf=5, max_features=.5, max_samples=.8, n_jobs=4, random_state=original["seed"]))]
        best = None
        for name, model in [("multiclass_boosting", base.model), *alternatives]:
            if model is not base.model:
                print(f"Fitting selection candidate: {name}", flush=True)
                model.fit(fit["X"], fit["y"], sample_weight=sample_weight)
            probabilities = model.predict_proba(selection["X"])
            background = int(np.flatnonzero(model.classes_ == "no_injection")[0])
            score = 1 - probabilities[:, background]
            macro = float(f1_score(selection["y"], model.classes_[probabilities.argmax(axis=1)],
                                   labels=list(FAULTS), average="macro", zero_division=0))
            pr_auc = float(average_precision_score(selection["y"] != "no_injection", score))
            objective = .6 * pr_auc + .4 * macro
            trial = {"model": name, "selection_detection_pr_auc": pr_auc,
                     "selection_macro_f1_faults": macro, "selection_objective": objective}
            trials.append(trial)
            print(trial, flush=True)
            if best is None or objective > best[0]:
                best = objective, name, model
        model = OperationalPatternModel(best[2], base.features)
        raw = model.model.predict_proba(calibration["X"])
        mapping = {label: index for index, label in enumerate(model.classes)}
        model.temperature = fit_temperature(raw, [mapping[value] for value in calibration["y"]])
        result = model.predict_features(runner.feature_frame(calibration, calibration_meta))
        model.threshold, threshold_metrics = choose_threshold(calibration["y"] != "no_injection", result["scenario_score"])
        model.metadata = {**base.metadata, "selected_model": best[1], "selection": trials,
                          "selection_objective": "0.6 * detection PR-AUC + 0.4 * fault-type macro-F1; selection partition only"}
        output.mkdir()
        model.save(output / "pattern_model.joblib")
        selection_report = runner.score_dataset(model, selection, selection_meta)
        calibration_report = runner.score_dataset(model, calibration, calibration_meta)
    # The IF reference and operating points are unchanged; verify before copying.
    if sha256_file(source / "isolation_baseline.joblib") != original["isolation_baseline_sha256"]:
        raise ValueError("reference baseline differs from its source run")
    (output / "isolation_baseline.joblib").write_bytes((source / "isolation_baseline.joblib").read_bytes())
    freeze = {**original, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_fingerprints": runner.source_fingerprints(),
              "model_sha256": sha256_file(output / "pattern_model.joblib"),
              "temperature": model.temperature, "decision_threshold": model.threshold,
              "selection_trials": trials, "selected_model": best[1],
              "calibration_threshold_metrics": threshold_metrics,
              "additional_selection_seconds": time.perf_counter() - started,
              "parent_fitted_run": {"frozen_sha256": sha256_file(source / "frozen.json"),
                                    "model_sha256": original["model_sha256"]},
              "selection_script_sha256": sha256_file(Path(__file__)),
              "final_test_status_at_freeze": "unexamined"}
    runner.write(output / "frozen.json", freeze)
    runner.write(output / "selection.json", selection_report)
    runner.write(output / "calibration.json", calibration_report)
    runner.write(output / "metrics.json", {"schema_version": "sih_benchmark_v2", "policy": freeze["policy"],
        "status": "frozen_awaiting_final_test", "split_audit": freeze["split_audit"],
        "selection": selection_report, "calibration": calibration_report, "final_test": None,
        "disclaimer": "Synthetic software-pattern evidence only; real hardware-fault accuracy remains unknown"})
    print(f"Selected and frozen {best[1]} at {output}; final data remains unexamined.", flush=True)


if __name__ == "__main__":
    main()
