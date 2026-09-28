"""Real-time (streaming) inference test: replay a real station's test
observations one-by-one through StreamingDetector, exactly as an AWS would emit
them, and measure live detection + per-observation latency.
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from awsad.streaming import StreamingDetector

ART = Path(r"C:/Users/Admin/AppData/Local/Temp/rt_artifacts/artifacts")
_cand = [Path(r"C:/Users/Admin/AppData/Local/Temp/rt_data/test_labeled.parquet"),
         Path(r"C:/Users/Admin/AppData/Local/Temp/rt_data/awsad_bundle_v3/data/processed/test_labeled.parquet")]
TEST = next(p for p in _cand if p.exists())

def main():
    fm = json.loads((ART / "feature_meta.json").read_text())
    fit_groups = set(fm["groups"])
    df = pd.read_parquet(TEST, columns=["timestamp", "station_id", "source",
        "temperature_c", "pressure_hpa", "relative_humidity_pct", "label"])
    df["group"] = df["station_id"].astype(str) + "|" + df["source"].astype(str)
    # pick a NOAA group present in the trained artifacts with the most labeled anomalies
    cand = (df[df["group"].isin(fit_groups) & df["group"].str.endswith("|noaa_isd")]
            .groupby("group")["label"].agg(["sum", "count"]))
    cand = cand[cand["count"] >= 5000].sort_values("sum", ascending=False)
    group = cand.index[0]
    g = df[df["group"] == group].sort_values("timestamp").reset_index(drop=True)
    print(f"streaming station group: {group}  ({len(g):,} hourly obs, "
          f"{int(g['label'].sum())} labelled-anomaly points)")

    det = StreamingDetector(group, ART)
    y = g["label"].to_numpy(dtype=int)
    pred = np.zeros(len(g), dtype=int); lat = np.empty(len(g))
    first_hits = []
    for i, row in enumerate(g.itertuples(index=False)):
        obs = {"timestamp": pd.Timestamp(row.timestamp),
               "temperature_c": row.temperature_c,
               "pressure_hpa": row.pressure_hpa,
               "relative_humidity_pct": row.relative_humidity_pct}
        out = det.update(obs)
        pred[i] = int(out["is_anomaly"]); lat[i] = out["latency_us"]
        if out["is_anomaly"] and y[i] == 1 and len(first_hits) < 3:
            first_hits.append((str(obs["timestamp"]), round(out["score"], 2),
                               round(out["confidence"], 2), out["top_channel"],
                               dict(out["qc_flags"])))

    warm = 200
    la = lat[warm:]
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    # event-level: fraction of labelled anomaly runs with >=1 live detection
    runs, i = [], 0
    while i < len(y):
        if y[i] == 1:
            j = i
            while j < len(y) and y[j] == 1: j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    caught = sum(1 for a, b in runs if pred[a:b].any())
    print(f"\n-- REAL-TIME RESULTS --")
    print(f"latency/obs: mean={la.mean():.1f}us  p95={np.quantile(la,0.95):.1f}us  "
          f"throughput={1e6/la.mean():,.0f} obs/s")
    print(f"point-wise: precision={prec:.3f}  recall={rec:.3f}  (TP={tp} FP={fp} FN={fn})")
    print(f"event-level: {caught}/{len(runs)} labelled anomaly episodes caught live "
          f"({caught/max(len(runs),1):.0%})")
    print("example live detections (ts, score, conf, top_channel, qc_flags):")
    for h in first_hits: print("  ", h)

if __name__ == "__main__":
    main()
