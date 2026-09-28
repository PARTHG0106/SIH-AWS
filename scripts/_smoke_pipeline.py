"""Local CPU smoke of run_pipeline on tiny synthetic data — validates the full
code path (incl. the new dropout QC flags) runs clean before a GPU retrain."""
import sys, json, tempfile, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from awsad.preprocessing.features import add_physics_derived
RNG = np.random.default_rng(7)
MEAS = ["temperature_c", "pressure_hpa", "relative_humidity_pct"]

def split_frame(sid, start, n, labeled):
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    t = 20 + 8*np.sin(np.arange(n)*2*np.pi/24) + RNG.normal(0, 1, n)
    p = 1000 + RNG.normal(0, 3, n)
    rh = (60 + 20*np.sin(np.arange(n)*2*np.pi/24 + 1) + RNG.normal(0, 5, n)).clip(1, 100)
    df = pd.DataFrame({"timestamp": ts, "station_id": sid, "source": "noaa_isd",
                       "temperature_c": t, "pressure_hpa": p, "relative_humidity_pct": rh})
    df = add_physics_derived(df)
    for c in MEAS:
        df[f"{c}__was_missing"] = np.float32(0)
    ev = []
    if labeled:
        lab = np.zeros(n, np.int8)
        for k, fault in enumerate(["spike", "dropout", "stuck"]):
            s0 = 200 + k*400; e0 = s0 + 25; lab[s0:e0] = 1
            if fault == "dropout":
                df.loc[s0:e0-1, "temperature_c"] = 0.0     # zero-fill dropout
            ev.append({"station_id": sid, "source": "noaa_isd", "fault": fault,
                       "channel": "temperature_c", "start": ts[s0], "end": ts[e0-1]})
        df["label"] = lab
        df["label_source"] = np.where(lab == 1, "synthetic", "")
    return df, ev

def main():
    from awsad.train_pipeline import load_processed, split_groups, run_pipeline
    p = Path(tempfile.mkdtemp())
    rows = {"train": [], "val": [], "test": []}; allev = []
    for sid, ho in [("42181099999", False), ("42182099999", False), ("43003099999", True)]:
        tr, _ = split_frame(sid, "2015-01-01", 2600, False)
        va, ev = split_frame(sid, "2023-01-01", 2600, True); allev += [{**e, "split": "val"} for e in ev]
        te, ev = split_frame(sid, "2024-01-01", 2600, True); allev += [{**e, "split": "test"} for e in ev]
        rows["train"].append(tr); rows["val"].append(va); rows["test"].append(te)
    for s, f in [("train", "train_clean"), ("val", "val_labeled"), ("test", "test_labeled")]:
        pd.concat(rows[s], ignore_index=True).to_parquet(p/f"{f}.parquet")
    pd.DataFrame(allev).to_parquet(p/"injection_events.parquet")
    (p/"splits.json").write_text(json.dumps({"build_id":"smoke","parser_version":"isd-qc-anomaly-labels-v3","holdout_stations":["43003099999"]}))
    for fn in ("station_stats","source_metadata","station_catalog"):
        (p/f"{fn}.json").write_text(json.dumps({"build_id":"smoke","parser_version":"isd-qc-anomaly-labels-v3","status":"fresh_online_builder","stations":[]}))
    dfs = load_processed(p); groups = split_groups(dfs)
    cfg = {n: {"epochs":1,"batch_size":64,"windows_per_epoch":128,"val_windows":128,
               "hidden":16,"layers":1,"dropout":0.0} for n in ("lstm_ae_24","lstm_ae_168","forecaster")}
    cfg["tx_ae"] = {"epochs":1,"batch_size":64,"windows_per_epoch":128,"val_windows":128,
                    "d_model":32,"nhead":4,"layers":1,"ff":64,"bottleneck":16,"dropout":0.0}
    out = Path(tempfile.mkdtemp())
    rep = run_pipeline(dfs, groups, out, train_deep=True, deep_cfg=cfg, log=print)
    print("=== PIPELINE OK ===  test keys:", list(rep.get("test", {}).keys())[:6])
    print("per_fault:", rep["test"].get("per_fault"))

if __name__ == "__main__":
    main()
