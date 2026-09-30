"""Supervised fault-TYPE classifier over the synthetic anomaly-injection benchmark.

Trains on the frozen detector's signals for injected-into-real-SURFRAD rows to name
the fault (spike / stuck / drift / …) — the "root-cause classification" SIH expects —
and reports a held-out confusion matrix + macro-F1 + feature importances. This is a
SYNTHETIC-label capability, kept separate from the unlabeled real-observation path.
Explainability is transparent: global permutation importances here, per-anomaly
signal-to-threshold attribution from the detector's own reason codes on the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from awsad.minute_detection import CHANNELS, SIGNALS
from awsad.preprocessing.anomaly_injection import AnomalyInjector, FAULT_WEIGHTS
from awsad.benchmark.injection_eval import FAULTS, _score, load_detector
from pathlib import Path

FEATURE_NAMES = [f"{c}__{s}" for c in CHANNELS for s in SIGNALS]


def _injected_table(cfg, models, thresholds, cache_dir, seed, target_fraction):
    X, y = [], []
    for path in sorted(Path(cache_dir).glob("*.parquet")):
        frame = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        injector = AnomalyInjector(rng=np.random.default_rng(seed), target_fraction=target_fraction)
        corrupt, _labels, events = injector.inject(frame, list(CHANNELS), required_faults=FAULTS)
        scored = _score(corrupt, models, thresholds, cfg)
        feats = scored[FEATURE_NAMES].to_numpy(float)  # HGB handles NaN natively
        fault_of = np.array(["normal"] * len(scored), dtype=object)
        stamp = corrupt["timestamp"]
        for _, e in events.iterrows():
            mask = (stamp.ge(e["start"]) & stamp.le(e["end"])).to_numpy()
            fault_of[mask] = e["fault"]
        # keep every injected row + a matched sample of normal rows for the NORMAL class
        faulted = fault_of != "normal"
        normal_idx = np.flatnonzero(~faulted)
        rng = np.random.default_rng(seed)
        keep_normal = rng.choice(normal_idx, size=min(len(normal_idx), int(faulted.sum())), replace=False)
        keep = np.union1d(np.flatnonzero(faulted), keep_normal)
        X.append(feats[keep]); y.append(fault_of[keep])
    return np.vstack(X), np.concatenate(y)


def train_fault_classifier(artifact_dir, cache_dir, *, seed: int = 26073, target_fraction: float = 0.08) -> dict:
    cfg, models, thresholds = load_detector(artifact_dir)
    X, y = _injected_table(cfg, models, thresholds, cache_dir, seed, target_fraction)
    labels = sorted(set(y))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    fault_labels = [l for l in labels if l != "normal"]
    report = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
    imp = permutation_importance(clf, Xte, yte, n_repeats=5, random_state=seed, n_jobs=1)
    importances = sorted(({"feature": f, "importance": float(v)}
                          for f, v in zip(FEATURE_NAMES, imp.importances_mean)),
                         key=lambda d: -d["importance"])
    shap_global = _shap_global(clf, Xte, seed)
    return {"policy": "supervised_fault_type_on_synthetic_injection", "n_train": int(len(ytr)),
            "n_test": int(len(yte)), "labels": labels,
            "macro_f1_faults": float(f1_score(yte, pred, average="macro", labels=fault_labels, zero_division=0)),
            "accuracy": float((pred == yte).mean()),
            "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
            "per_class": {k: report[k] for k in labels if k in report},
            "feature_importances": importances, "shap_mean_abs": shap_global,
            "explainability": "permutation importances + SHAP mean|value| per feature (global); "
                              "per-anomaly attribution uses the detector's own signal-to-threshold reason codes"}


def _shap_global(clf, X, seed):
    """Best-effort global SHAP: mean |SHAP value| per feature. Returns None if unavailable."""
    try:
        import shap
        rng = np.random.default_rng(seed)
        sample = X[rng.choice(len(X), min(300, len(X)), replace=False)]
        values = shap.TreeExplainer(clf).shap_values(sample)
        arr = np.abs(np.asarray(values))
        feature_axis = int(np.argmax([s == len(FEATURE_NAMES) for s in arr.shape]))
        mean_abs = arr.mean(axis=tuple(a for a in range(arr.ndim) if a != feature_axis))
        return sorted(({"feature": f, "mean_abs_shap": float(v)} for f, v in zip(FEATURE_NAMES, mean_abs)),
                      key=lambda d: -d["mean_abs_shap"])
    except Exception as exc:  # noqa: BLE001 - explainability is best-effort
        return {"unavailable": str(exc)}
