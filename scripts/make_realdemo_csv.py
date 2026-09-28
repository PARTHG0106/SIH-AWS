"""Extract a real labeled test slice (with injected faults) for an end-to-end
inference demo on authentic NOAA ISD data."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awsad.train_pipeline import load_processed, _get_group  # noqa: E402

G = "42182099999|noaa_isd"
dfs = load_processed("data/processed")
te = _get_group(dfs["test"], G).reset_index(drop=True)
ev = dfs["events"]

events = ev[(ev.station_id.astype(str) == "42182099999") & (ev.source == "noaa_isd")
            & (ev["split"] == "test")].sort_values("start").reset_index(drop=True)
print(events[["fault", "channel", "start", "end"]].head(20).to_string())

best = None
for i in range(len(events) - 2):
    a, b = events.iloc[i], events.iloc[min(i + 2, len(events) - 1)]
    span = (b["end"] - a["start"]).total_seconds() / 3600
    if span < 24 * 45:
        best = (a["start"], b["end"])
        break
if best is None:
    best = (events.iloc[0]["start"], events.iloc[0]["end"])

m = (te.timestamp >= best[0] - pd.Timedelta(days=2)) & \
    (te.timestamp <= best[1] + pd.Timedelta(days=2))
sub = te[m].reset_index(drop=True)
cols = ["timestamp", "station_id", "temperature_c", "pressure_hpa",
        "relative_humidity_pct", "label"]
out = sub[cols].rename(columns={"label": "true_label"})
out.to_csv("scripts/real_demo.csv", index=False)
print(f"\nwindow {best[0]} .. {best[1]}  rows={len(out)}  "
      f"labeled={int(out.true_label.sum())}")
print("wrote scripts/real_demo.csv")
