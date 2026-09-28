import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from awsad.streaming import StreamingDetector

d = StreamingDetector("42182099999|noaa_isd", "artifacts_full_classical")
for c in ("temperature_c", "pressure_hpa", "relative_humidity_pct"):
    print(c, "clim entries:", len(d.climatology.get(c, {}).get("table", {})),
          "sigma:", round(d.params[c]["sigma"], 3))

rng = np.random.default_rng(1)
outs = []
for i in range(300):
    outs.append(d.update({
        "timestamp": pd.Timestamp("2024-06-01", tz="UTC") + pd.Timedelta(hours=i),
        "temperature_c": 30 + rng.normal(),
        "pressure_hpa": 1009 + rng.normal() * 0.2,
        "relative_humidity_pct": 50 + rng.normal()}))
print("stream ok; mean score", round(float(np.mean([o["score"] for o in outs])), 4),
      "flags", sum(o["is_anomaly"] for o in outs))

# inject a stuck fault mid-stream and confirm escalation
for i in range(40):
    outs.append(d.update({
        "timestamp": pd.Timestamp("2024-06-14", tz="UTC") + pd.Timedelta(hours=i),
        "temperature_c": 33.33, "pressure_hpa": 1008.5,
        "relative_humidity_pct": 61.0}))
print("stuck-temperature test -> is_anomaly:", outs[-1]["is_anomaly"],
      "| score:", round(outs[-1]["score"], 3),
      "| flags:", outs[-1]["qc_flags"])
