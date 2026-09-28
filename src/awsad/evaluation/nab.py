"""Numenta Anomaly Benchmark (NAB) evaluation adapter.

Our weather models are multivariate; NAB series are univariate. For an honest
external benchmark we run a small self-contained univariate detector
(Isolation Forest on sliding-window shape features) against the labeled NAB
corpus and report point-adjusted F1. Corpus path must already exist offline
(it is shipped inside the Kaggle dataset bundle under nab/).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .metrics import point_adjust

NAB_SUBSETS = [
    "realKnownCause/ambient_temperature_system_failure.csv",
    "realKnownCause/machine_temperature_system_failure.csv",
    "realKnownCause/nyc_taxi.csv",
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv",
]


def _window_features(x: np.ndarray, w: int = 48) -> np.ndarray:
    s = pd.Series(x, dtype=float)
    feats = pd.DataFrame({
        "v": s,
        "d1": s.diff(),
        "std": s.rolling(w, min_periods=4).std(),
        "roll_med": s.rolling(w, min_periods=4).median(),
        "roll_mean": s.rolling(w, min_periods=4).mean(),
        "ewm": s.ewm(halflife=w / 4).mean(),
    })
    return feats.fillna(0).to_numpy(dtype=np.float32)


def evaluate_nab(corpus_dir: Path, subsets: list[str] | None = None,
                 contamination: float = 0.02, seed: int = 42) -> pd.DataFrame:
    corpus_dir = Path(corpus_dir)
    with open(corpus_dir / "labels" / "combined_windows.json") as f:
        windows = json.load(f)

    rows = []
    for rel in (subsets or NAB_SUBSETS):
        fp = corpus_dir / "data" / rel
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        x = df["value"].to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=np.nanmedian(x))
        X = _window_features(x)
        split = max(200, int(0.3 * len(x)))
        clf = IsolationForest(n_estimators=200, contamination=contamination,
                              random_state=seed, n_jobs=-1)
        clf.fit(X[:split])
        scores = -clf.decision_function(X)
        y = pd.Series(0, index=df.index)
        for a, b in windows.get(rel, []):
            mask = (df["timestamp"] >= a) & (df["timestamp"] <= b)
            y[mask.to_numpy()] = 1
        # Calibrate only on the same leading prefix used to fit the detector.
        # Taking a quantile over the full series would adapt the threshold to
        # the benchmark anomalies being evaluated.
        thr = np.quantile(scores[:split], 1 - contamination)
        m = point_adjust(y.to_numpy(), (scores >= thr).astype(int))
        rows.append({"series": Path(rel).name, "n": len(x),
                     "n_windows": len(windows.get(rel, [])), **m})
    return pd.DataFrame(rows)
