"""Train + evaluate the supervised fault-TYPE classifier on the synthetic injection
benchmark and write a clearly-labelled artifact (confusion matrix, per-class metrics,
permutation importances). Root-cause classification is a SYNTHETIC-label capability.

    python scripts/run_fault_classifier.py [artifact_dir] [cache_dir]
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")

from awsad.benchmark.fault_classifier import train_fault_classifier
from awsad.minute_detection import write_json


def main() -> None:
    artifact_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts_minute_20260928")
    cache_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/native_minute_fresh_20260928")
    out = artifact_dir / "injection_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    result = train_fault_classifier(artifact_dir, cache_dir, model_out=out / "fault_classifier.joblib")
    write_json(out / "classifier.json", result)
    print(f"macro-F1(faults)={result['macro_f1_faults']:.3f} accuracy={result['accuracy']:.3f} "
          f"train={result['n_train']} test={result['n_test']}")
    print(f"wrote {out / 'classifier.json'}")


if __name__ == "__main__":
    main()
