"""Run the synthetic anomaly-injection benchmark against a frozen minute detector
and write a clearly-labelled artifact. SIH is judged on anomaly-injected data; this
injects into HELD-OUT real SURFRAD and scores with the frozen detector (no retrain).

    python scripts/run_injection_benchmark.py [artifact_dir] [cache_dir]

Defaults: artifacts_minute_20260928 + data/native_minute_fresh_20260928 (Jan-2025
holdout month). Output: <artifact_dir>/injection_benchmark/metrics.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from awsad.benchmark.injection_eval import run_injection_benchmark
from awsad.minute_detection import write_json


def main() -> None:
    artifact_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts_minute_20260928")
    cache_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/native_minute_fresh_20260928")
    result = run_injection_benchmark(artifact_dir, cache_dir)
    out = artifact_dir / "injection_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", result)
    p = result["point"]
    print(f"point F1={p['f1']:.3f} P={p['precision']:.3f} R={p['recall']:.3f} ROC-AUC={p['roc_auc']}")
    print(f"clean candidate rate={result['clean_candidate_rate']:.4f}")
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
