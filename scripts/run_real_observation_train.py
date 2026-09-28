"""Run a verified real-observation forecasting experiment on CPU.

Examples:
  python scripts/run_real_observation_train.py --dataset-dir data/real_surfrad --out artifacts_real/selection --selection-only --holdout-station gwn --train-end 2024-09-01T00:00:00Z --validation-end 2024-11-01T00:00:00Z

The registered source verifier checks raw/processed equality before model imports. Initial model
comparisons should use --selection-only; make the final test run with the chosen
configuration in a new output directory. Unknown faults are never labelled normal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awsad.data.training_contract import TrainingDataContractError, require_eligible_training_data


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="verified bundle containing observations.parquet and source_metadata.json")
    parser.add_argument("--out", type=Path, required=True, help="new experiment directory outside the dataset")
    parser.add_argument("--config", type=Path, help="JSON fields of RealObservationConfig; command options override it")
    parser.add_argument("--preflight-only", action="store_true", help="verify input without loading observations or fitting models")
    test_mode = parser.add_mutually_exclusive_group()
    test_mode.add_argument("--selection-only", action="store_true", help="do not inspect test target values or QC")
    test_mode.add_argument("--evaluate-test", action="store_true", help="evaluate the final configuration on the reserved test period")
    parser.add_argument("--horizon-hours", type=float, nargs="+", help="actual forecast horizon(s), such as 1 or 1 24")
    parser.add_argument("--holdout-station", action="append", dest="holdout_station_ids",
                        help="station excluded from fitting and calibration; repeat for multiple IDs")
    parser.add_argument("--train-end", help="exclusive UTC training boundary, with timezone")
    parser.add_argument("--validation-end", help="exclusive UTC validation boundary, with timezone")
    parser.add_argument("--test-end", help="exclusive UTC test boundary, with timezone")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--min-train-rows", type=int)
    parser.add_argument("--min-validation-rows", type=int)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--max-cpu-threads", type=int)
    parser.add_argument("--max-leaf-nodes", type=int)
    parser.add_argument("--min-samples-leaf", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--l2-regularization", type=float)
    parser.add_argument("--residual-shrinkage", type=float)
    parser.add_argument("--anomaly-quantile", type=float)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--thinning-minutes", type=int, help="select first existing report in each clock interval")
    parser.add_argument("--all-native-rows", action="store_true", help="disable actual-record thinning")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    dataset = args.dataset_dir.resolve()
    output = args.out.resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("experiment outputs must be outside the immutable input dataset")
    if (output / "metrics.json").exists():
        raise FileExistsError("use a new output directory to preserve prior experiment evidence")
    try:
        preflight = require_eligible_training_data(dataset, report_path=output / "training_preflight.json")
    except TrainingDataContractError as exc:
        print(str(exc), file=sys.stderr)
        print("Input was refused; no forecasting model was trained.")
        print("Preflight report:", output / "training_preflight.json")
        return 2
    if args.preflight_only:
        print("Raw and processed observation values verified. No model was trained.")
        print("Preflight report:", output / "training_preflight.json")
        return 0

    from dataclasses import fields
    import pandas as pd
    from awsad.real_observation_pipeline import RealObservationConfig, run_real_observation_pipeline

    payload = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a JSON object")
    valid_fields = {field.name for field in fields(RealObservationConfig)}
    unknown = set(payload) - valid_fields
    if unknown:
        raise ValueError("unknown configuration fields: " + repr(sorted(unknown)))
    for name in valid_fields:
        if name == "evaluate_test":
            continue  # Only explicit mutually-exclusive flags override JSON/default.
        value = getattr(args, name, None)
        if value is not None:
            payload[name] = value
    if args.horizon_hours is not None:
        minutes = [hours * 60 for hours in args.horizon_hours]
        if any(value <= 0 or not value.is_integer() for value in minutes):
            raise ValueError("horizons must resolve to a positive integer number of minutes")
        payload["horizon_minutes"] = tuple(int(value) for value in minutes)
    if args.selection_only:
        payload["evaluate_test"] = False
    if args.evaluate_test:
        payload["evaluate_test"] = True
    if args.all_native_rows:
        if args.thinning_minutes is not None:
            raise ValueError("choose --all-native-rows or --thinning-minutes, not both")
        payload["thinning_minutes"] = None
    for name in ("horizon_minutes", "context_lags_minutes", "holdout_station_ids"):
        if payload.get(name) is not None:
            payload[name] = tuple(payload[name])
    config = RealObservationConfig(**payload)
    config.validate()
    metadata = json.loads((dataset / "source_metadata.json").read_text(encoding="utf-8"))
    observations = pd.read_parquet(dataset / "observations.parquet")
    print("Verified original observations:", len(observations))
    print("Experiment mode:", "final test evaluation" if config.evaluate_test else "validation selection only")
    report = run_real_observation_pipeline(observations, output, config, source_metadata=metadata)
    source_paths = [ROOT / "src/awsad/real_observation_pipeline.py",
                    ROOT / "src/awsad/data/training_contract.py", Path(__file__).resolve()]
    run_manifest = {"dataset_directory": str(dataset), "preflight": preflight,
                    "training_source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in source_paths},
                    "data_fingerprint": report["data_fingerprint"], "config": report["config"]}
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print("Experiment status:", report["status"])
    print("Models fitted:", report["trained_channel_models"], "/", report["expected_channel_models"])
    print("Forecast and score evidence:", output / "metrics.json")
    print("Observed records with separate predictions:", output / "scored_observations.parquet")
    return 0 if report["trained_channel_models"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
