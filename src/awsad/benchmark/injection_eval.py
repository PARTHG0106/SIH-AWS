"""Anomaly-injection benchmark for the frozen minute detector (SIH Detection-Accuracy).

SIH26073 is judged "in anomaly injected data": we inject the nine-fault taxonomy
(spike/drift/bias/stuck/dropout/noise_burst/clipping/scale_error/sensor_swap) into
HELD-OUT real SURFRAD observations, score them with the FROZEN detector (never
retraining or recalibrating), and measure detection against the known injection
labels. These are clearly-labelled SYNTHETIC-injection results, reported separately
from the real-observation replay — real observations still carry no fabricated labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from awsad.minute_detection import (CHANNELS, MinuteConfig, apply_thresholds, compute_signals)
from awsad.preprocessing.anomaly_injection import AnomalyInjector, FAULT_WEIGHTS, INJECTOR_VERSION
from awsad.benchmark.consistency import physical_flags

FAULTS = tuple(FAULT_WEIGHTS)


def load_detector(artifact_dir: str | Path):
    art = Path(artifact_dir)
    raw = json.loads((art / "config.json").read_text())
    cfg = MinuteConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()})
    models = joblib.load(art / "models.joblib")
    thresholds = json.loads((art / "detector.json").read_text())["thresholds"]
    return cfg, models, thresholds


def _score(frame, models, thresholds, cfg):
    signals, _elig, predictions, _feats = compute_signals(frame, models, cfg)
    return apply_thresholds(frame, signals, predictions, thresholds, cfg)


def _prf(truth: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
# __EVAL_TAIL__


def run_injection_benchmark(artifact_dir, cache_dir, *, seed: int = 26073,
                            target_fraction: float = 0.06) -> dict:
    cfg, models, thresholds = load_detector(artifact_dir)
    files = sorted(Path(cache_dir).glob("*.parquet"))
    if not files:
        raise ValueError(f"no station parquet files under {cache_dir}")
    truth_all, pred_all, predc_all, score_all, clean_rates, clean_comb = [], [], [], [], [], []
    row_hits = {f: [0, 0] for f in FAULTS}      # [caught_rows, total_rows] detector only
    row_hits_c = {f: [0, 0] for f in FAULTS}    # [caught_rows, total_rows] detector + physical
    ev_hits = {f: [0, 0] for f in FAULTS}        # [caught_events, total_events]
    latency = {f: [] for f in FAULTS}
    for path in files:
        frame = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        clean = _score(frame, models, thresholds, cfg)
        clean_det = clean["is_candidate"].to_numpy(dtype=bool)
        clean_rates.append(float(clean_det.mean()))
        clean_comb.append(float((clean_det | physical_flags(frame)["any"]).mean()))
        injector = AnomalyInjector(rng=np.random.default_rng(seed), target_fraction=target_fraction)
        corrupt, labels, events = injector.inject(frame, list(CHANNELS), required_faults=FAULTS)
        scored = _score(corrupt, models, thresholds, cfg)
        truth = labels.to_numpy().astype(bool)
        pred = scored["is_candidate"].to_numpy(dtype=bool)
        predc = pred | physical_flags(corrupt)["any"]
        truth_all.append(truth); pred_all.append(pred); predc_all.append(predc)
        score_all.append(np.nan_to_num(scored["anomaly_score"].to_numpy(float), nan=0.0))
        stamp = corrupt["timestamp"]
        for _, e in events.iterrows():
            fault = e["fault"]
            mask = (stamp.ge(e["start"]) & stamp.le(e["end"])).to_numpy()
            if not mask.any():
                continue
            row_hits[fault][1] += int(mask.sum()); row_hits[fault][0] += int((mask & pred).sum())
            row_hits_c[fault][1] += int(mask.sum()); row_hits_c[fault][0] += int((mask & predc).sum())
            ev_hits[fault][1] += 1
            caught = mask & predc
            if caught.any():
                ev_hits[fault][0] += 1
                latency[fault].append((stamp[caught].min() - stamp[mask].min()).total_seconds() / 60.0)
    truth = np.concatenate(truth_all); pred = np.concatenate(pred_all)
    predc = np.concatenate(predc_all); score = np.concatenate(score_all)
    point = _prf(truth, pred)
    point["roc_auc"] = (float(roc_auc_score(truth, score)) if truth.any() and (~truth).any() else None)
    combined = _prf(truth, predc)
    per_fault = {f: {"rows": row_hits[f][1],
                     "row_recall": (row_hits[f][0] / row_hits[f][1] if row_hits[f][1] else None),
                     "row_recall_combined": (row_hits_c[f][0] / row_hits_c[f][1] if row_hits_c[f][1] else None),
                     "events": ev_hits[f][1], "event_recall": (ev_hits[f][0] / ev_hits[f][1] if ev_hits[f][1] else None),
                     "median_latency_min": (float(np.median(latency[f])) if latency[f] else None)} for f in FAULTS}
    return {"injector_version": INJECTOR_VERSION, "policy": "synthetic_injection_into_real_holdout_frozen_detector",
            "disclaimer": "Synthetic injected faults on real SURFRAD; not real hardware-fault labels or IMD accuracy.",
            "seed": seed, "target_fraction": target_fraction, "stations": [p.stem for p in files],
            "clean_candidate_rate": (float(np.mean(clean_rates)) if clean_rates else None),
            "clean_candidate_rate_combined": (float(np.mean(clean_comb)) if clean_comb else None),
            "point": point, "point_combined": combined, "per_fault": per_fault,
            "point_semantics": "detector is_candidate (+physical for *_combined) vs any-injected-fault row; ROC-AUC over anomaly_score"}
