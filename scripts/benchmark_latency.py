"""Edge/real-time latency benchmark for the streaming detector.

    python scripts/benchmark_latency.py --artifacts artifacts_full_classical --rows 20000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awsad.streaming import StreamingDetector, benchmark_latency  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts_full_classical")
    ap.add_argument("--group", default=None)
    ap.add_argument("--rows", type=int, default=20000)
    args = ap.parse_args()

    groups = list((__import__("json").loads(
        (Path(args.artifacts) / "ensemble.json").read_text())
        .get("per_group_norms", {}) or {}).keys())
    group = args.group or (groups[0] if groups else None)
    if not group:
        raise SystemExit("no groups in artifacts — train first")

    det = StreamingDetector(group, args.artifacts)
    clim = det.climatology
    rng = np.random.default_rng(0)
    base = {c: (clim.get(c, {}).get("fallback")
                or {"temperature_c": 30.0, "pressure_hpa": 1010.0,
                    "relative_humidity_pct": 60.0}[c])
            for c in ("temperature_c", "pressure_hpa", "relative_humidity_pct")}
    start = pd.Timestamp.now(tz="UTC").floor("h")
    rows = [{"timestamp": start + pd.Timedelta(hours=i),
             "temperature_c": base["temperature_c"] + rng.normal(0, 0.5),
             "pressure_hpa": base["pressure_hpa"] + rng.normal(0, 0.2),
             "relative_humidity_pct": base["relative_humidity_pct"] + rng.normal(0, 2)}
            for i in range(args.rows)]

    rep = benchmark_latency(det, rows)
    print(f"Streaming benchmark on {group}: "
          f"{rep['n']} obs | mean {rep['mean_us']:.1f} us  p95 {rep['p95_us']:.0f} us  "
          f"throughput {rep['throughput_obs_per_s']:,.0f} obs/s")


if __name__ == "__main__":
    main()
