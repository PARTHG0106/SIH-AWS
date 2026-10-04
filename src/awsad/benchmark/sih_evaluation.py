"""Unadjusted scenario metrics, calibration diagnostics and method baselines."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)


def point_metrics(truth, predicted, score=None):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    result = {"precision": precision, "recall": recall,
              "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
              "tp": tp, "fp": fp, "fn": fn, "tn": tn, "rows": len(truth),
              "unmodified_row_candidate_rate": fp / (fp + tn) if fp + tn else None}
    if score is not None and truth.any() and (~truth).any():
        result.update(roc_auc=float(roc_auc_score(truth, score)),
                      pr_auc=float(average_precision_score(truth, score)))
    return result


def choose_threshold(truth, scores, *, max_background_rate=.02):
    truth, scores = np.asarray(truth, bool), np.asarray(scores, float)
    candidates = np.unique(np.r_[np.linspace(.05, .99, 95), np.quantile(scores, [.95, .98, .99, .995, .999]), 1.000001])
    rows = []
    for threshold in candidates:
        metrics = point_metrics(truth, scores >= threshold)
        if metrics["unmodified_row_candidate_rate"] <= max_background_rate:
            rows.append((metrics["f1"], metrics["recall"], -float(threshold), metrics))
    selected = max(rows, key=lambda row: row[:3])
    return -selected[2], selected[3]


def reliability(truth, probabilities, bins=10):
    truth = np.asarray(truth, float)
    probabilities = np.clip(np.asarray(probabilities, float), 0, 1)
    result = []
    ece = 0.
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (probabilities >= low) & ((probabilities < high) if index + 1 < bins else (probabilities <= high))
        count = int(selected.sum())
        if not count:
            continue
        predicted = float(probabilities[selected].mean())
        observed = float(truth[selected].mean())
        ece += count / len(truth) * abs(predicted - observed)
        result.append({"lower": low, "upper": high, "rows": count,
                       "predicted_frequency": predicted, "observed_modification_frequency": observed})
    return {"brier": float(np.square(probabilities - truth).mean()), "ece": float(ece),
            "bins": result, "semantics": "synthetic modification frequency, not real hardware failure probability"}


def evaluate(labels, predicted, scores, predicted_types, scenario_indices, positions, events):
    labels = np.asarray(labels)
    truth = labels != "no_injection"
    predicted = np.asarray(predicted, bool)
    scores = np.asarray(scores, float)
    positions = np.asarray(positions)
    scenario_indices = np.asarray(scenario_indices)
    faults = sorted({event["fault"] for event in events if event["fault"] != "no_injection"})
    per_fault = {}
    event_records = []
    background = np.zeros(len(labels), bool)
    for index, event in enumerate(events):
        selected = scenario_indices == index
        if event["fault"] == "no_injection":
            background |= selected
            continue
        changed = selected & truth
        if not changed.any():
            event_records.append({"scenario_id": event["scenario_id"], "fault": event["fault"],
                                  "applied": False, "detected": None, "latency_minutes": None})
            continue
        hit = changed & predicted
        delay = int(positions[hit].min() - positions[changed].min()) if hit.any() else None
        event_records.append({"scenario_id": event["scenario_id"], "fault": event["fault"],
                              "applied": True, "detected": bool(hit.any()), "latency_minutes": delay})
    for fault in faults:
        mask = labels == fault
        rows = [row for row in event_records if row["fault"] == fault and row["applied"]]
        delays = [row["latency_minutes"] for row in rows if row["detected"]]
        per_fault[fault] = {"modified_rows": int(mask.sum()),
                            "row_recall": float(predicted[mask].mean()) if mask.any() else None,
                            "events": len(rows), "detected_events": sum(row["detected"] for row in rows),
                            "event_recall": sum(row["detected"] for row in rows) / len(rows) if rows else None,
                            "median_latency_min": float(np.median(delays)) if delays else None,
                            "p95_latency_min": float(np.quantile(delays, .95)) if delays else None,
                            "not_applied_events": sum(row["fault"] == fault and not row["applied"] for row in event_records)}
    result = {"point": point_metrics(truth, predicted, scores), "per_fault": per_fault,
              "event_records": event_records, "reliability": reliability(truth, scores),
              "unmodified_baseline_candidate_rate": float(predicted[background].mean()) if background.any() else None,
              "semantics": "strict row and modified-interval event metrics against synthetic interventions; no point adjustment; no real hardware truth"}
    if predicted_types is not None:
        classes = ["no_injection", *faults]
        report = classification_report(labels, predicted_types, labels=classes, output_dict=True, zero_division=0)
        result["classification"] = {"labels": classes,
            "macro_f1_faults": float(f1_score(labels, predicted_types, labels=faults, average="macro", zero_division=0)),
            "accuracy": float(np.mean(labels == predicted_types)),
            "confusion_matrix": confusion_matrix(labels, predicted_types, labels=classes).tolist(),
            "per_class": {key: report[key] for key in classes}}
    return result


def simple_scores(features):
    """Independent transparent baseline methods, not copied competitor results."""
    def maximum(columns):
        values = features[columns].abs().to_numpy(float)
        return np.nan_to_num(values, nan=0.).max(axis=1)
    step = maximum([c for c in features if c.endswith(":delta_1")])
    zscore = maximum([c for c in features if c.endswith(":z_120")])
    missing = maximum([c for c in features if c.endswith(":missing")])
    return {"rolling_zscore": np.maximum(zscore, missing * 50),
            "step_change": np.maximum(step, missing * 50)}
