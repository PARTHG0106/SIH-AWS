"""Generate a small demo AWS CSV with obvious faults (for run_inference demo)."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 24 * 190
ts = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
diurnal = np.sin(2 * np.pi * ts.hour.to_numpy() / 24)
t = 28 + 9 * diurnal + rng.normal(0, 0.6, n)
p = 1013 + 4 * np.sin(2 * np.pi * ts.dayofyear.to_numpy() / 365) + rng.normal(0, 0.3, n)
rh = np.clip(58 - 16 * diurnal + rng.normal(0, 2.5, n), 1, 100)

# frozen temperature for 36 h; humidity drift for a week; two spikes in pressure
t[1500:1536] = 31.4
rh[2400:2568] = rh[2400:2568] * np.linspace(1.0, 1.35, 168)
p[3000] += 12
p[3010] += 15

df = pd.DataFrame({"timestamp": ts, "station_id": "42182099999",
                   "temperature_c": t.round(2), "pressure_hpa": p.round(2),
                   "relative_humidity_pct": rh.round(1)})
out = Path("scripts/demo_station.csv")
df.to_csv(out, index=False)
print(f"wrote {out} with {len(df)} rows")
