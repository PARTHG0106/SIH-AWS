"""Evaluation metrics for time-series anomaly detection.

Implements the three standard families so results are comparable with the
literature and defensible in front of judges:
  * pointwise P/R/F1 (strict)
  * point-adjusted P/R/F1 (Xu et al. 2018 — upper bound every paper reports)
  * range-wise P/R/F1 (Tatbul et al. 2018, NIPS paper "Precision and Recall
    for Time Series") — positional-bias "flat"

Also PR-AUC / ROC-AUC and mean detection latency.
"""
from __future__ import annotations

import numpy as np
from .thresholds import threshold_above


def _segments(labels: np.ndarray) -> list[tuple[int, int]]:
    # Unknown labels break events; they must never be treated as normal truth.
    idx = np.flatnonzero(np.diff(np.concatenate(([0], np.asarray(labels) == 1, [0]))))
    return [(int(idx[i]), int(idx[i + 1] - 1)) for i in range(0, len(idx), 2)]


def pointwise(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """If any point in a true segment is hit, the whole segment counts as TP."""
    adj = y_pred.copy()
    for s, e in _segments(y_true):
        if y_pred[s:e + 1].any():
            adj[s:e + 1] = 1
    return pointwise(y_true, adj)


def range_wise(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tatbul et al. 2018 with flat positional bias (gamma=1)."""
    true_segs = _segments(y_true)
    known = np.isin(y_true, [0, 1])
    pred_segs = _segments(np.where(known, y_pred, 0))

    if not pred_segs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Prefix counts preserve the existing alpha=1 existence-recall / flat
    # overlap-precision definitions without a quadratic event-pair loop.
    pred_count = np.r_[0, np.cumsum((np.asarray(y_pred) == 1) & known)]
    true_count = np.r_[0, np.cumsum(np.asarray(y_true) == 1)]
    recall = sum(pred_count[e + 1] > pred_count[s] for s, e in true_segs) / max(1, len(true_segs))
    prec_total = sum((true_count[e + 1] - true_count[s]) / (e - s + 1)
                     for s, e in pred_segs)
    precision = prec_total / max(1, len(pred_segs))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aucs(y_true: np.ndarray, scores: np.ndarray) -> dict:
    known = np.isin(y_true, [0, 1]) & np.isfinite(scores)
    y_true, scores = np.asarray(y_true)[known], np.asarray(scores)[known]
    if len(np.unique(y_true)) != 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        return {"pr_auc": float(average_precision_score(y_true, scores)),
                "roc_auc": float(roc_auc_score(y_true, scores))}
    except Exception:
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}


def mean_detection_latency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lats = []
    for s, e in _segments(y_true):
        hits = np.flatnonzero(y_pred[s:e + 1])
        if len(hits):
            lats.append(float(hits[0]))
    return float(np.mean(lats)) if lats else float("nan")


def best_weight_objective_over_quantiles(
        scores: np.ndarray, y_true: np.ndarray,
        quantiles: np.ndarray | None = None,
        strict_weight: float = 1.0,
        precision_floor: float = 0.35) -> tuple[float, float]:
    """Validation objective for ensemble-weight calibration.

    Strict pointwise F1 carries most of the weight. A smaller point-adjusted
    term rewards detection inside a fault interval without allowing the
    lenient metric to dominate model selection.
    """
    if strict_weight == 1.0 and quantiles is None:
        _, detail = choose_threshold(scores, y_true, precision_floor=precision_floor)
        return detail.get("objective", 0.0), detail.get("quantile", 1.0)
    known = np.isin(y_true, [0, 1]) & np.isfinite(scores)
    if not known.any():
        return 0.0, 1.0
    # Preserve unknown gaps for event diagnostics, exclude them from counts.
    y_true = np.where(known, y_true, np.nan)
    q = quantiles if quantiles is not None else np.linspace(0.90, 0.999, 25)
    true_segs = _segments(y_true)
    sorted_s = np.sort(scores[known])
    best_score, best_q = 0.0, float(q[-1])
    strict_weight = float(np.clip(strict_weight, 0.0, 1.0))
    for qi in q:
        thr = float(np.quantile(sorted_s, qi))
        pred = (scores >= thr) & known
        pw = pointwise(y_true, pred)
        tp_pred = 0.0
        n_pred = int(pred.sum())
        for s, e in true_segs:
            hit = pred[s:e + 1].any()
            if hit:
                tp_pred += (e - s + 1)
        # point-adjusted equivalents
        tp = float(tp_pred)
        fp = float(n_pred - sum(pred[s:e + 1].sum() for s, e in true_segs))
        fn = float(np.count_nonzero(y_true == 1) - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        pa_f1 = 2 * p * r / (p + r) if p + r else 0.0
        objective = strict_weight * pw["f1"] + (1.0 - strict_weight) * pa_f1
        if pw["precision"] < precision_floor:
            objective *= 0.5
        if objective > best_score:
            best_score, best_q = objective, qi
    return best_score, best_q


def best_pa_f1_over_quantiles(scores: np.ndarray, y_true: np.ndarray,
                              quantiles: np.ndarray | None = None) -> tuple[float, float]:
    """Backward-compatible pure point-adjusted objective."""
    return best_weight_objective_over_quantiles(
        scores, y_true, quantiles=quantiles, strict_weight=0.0,
        precision_floor=0.0)


def choose_threshold(scores: np.ndarray, y: np.ndarray,
                     quantiles: np.ndarray | None = None,
                     precision_floor: float = 0.35) -> tuple[float, dict]:
    """Maximise strict F1 over every distinct validation score in O(n log n).

    Unknown labels and unavailable scores are excluded. Ties are evaluated as
    whole blocks to match the deployed ``>=`` rule. Equal objectives prefer
    the higher threshold. An explicit no-alarm candidate avoids flagging every
    row of an all-negative or constant-score validation set. No test data are
    used. ``quantiles`` optionally restricts the candidate thresholds.
    """
    score_dtype = np.asarray(scores).dtype
    scores, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=float)
    if scores.ndim != 1 or y.shape != scores.shape:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    valid = np.isfinite(scores) & np.isin(y, [0, 1])
    n = int(valid.sum())
    if not n:
        finite = scores[np.isfinite(scores)]
        threshold = threshold_above(finite.max(), score_dtype) if len(finite) else 1e3
        return threshold, {"degenerate": True, "reason": "no_known_scored_labels",
                           "n_labeled": 0, "objective": 0.0, "quantile": 1.0}
    order = np.argsort(-scores[valid], kind="stable")
    s, truth = scores[valid][order], y[valid][order]
    end = np.r_[np.flatnonzero(s[:-1] != s[1:]), n - 1]
    thresholds = np.r_[threshold_above(s[0], score_dtype), s[end]]
    tp = np.r_[0, np.cumsum(truth == 1)[end]].astype(float)
    predicted = np.r_[0, end + 1]
    positives = int(np.count_nonzero(truth == 1))
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * tp, predicted + positives, out=np.zeros_like(tp),
                   where=(predicted + positives) > 0)
    objective = f1 * np.where(precision < precision_floor, 0.5, 1.0)
    if quantiles is not None:
        qs = np.asarray(quantiles, dtype=float)
        if not len(qs) or np.any((qs < 0) | (qs > 1)):
            raise ValueError("quantiles must be nonempty and in [0, 1]")
        candidates = np.quantile(s, qs)
        # A threshold between distinct scores is equivalent to the next larger
        # observed score. Include the no-alarm candidate for every search.
        selected = np.unique(np.r_[0, np.searchsorted(-s, -candidates, side="right")])
        allowed = np.isin(predicted, selected)
        objective = np.where(allowed, objective, -1.0)
    best = int(np.argmax(objective))
    threshold = float(thresholds[best])
    pred = valid & (scores >= threshold)
    masked_truth = np.where(valid, y, np.nan)
    return threshold, {
        "pw": pointwise(masked_truth, pred), "pa": point_adjust(masked_truth, pred),
        "objective": float(objective[best]), "quantile": float(1 - predicted[best] / n),
        "objective_weights": {"pointwise_f1": 1.0, "point_adjusted_f1": 0.0},
        "n_labeled": n, "n_excluded": int(len(y) - n),
        "n_candidates": int(len(thresholds)), "no_alarm": bool(best == 0),
        "degenerate": positives == 0 or positives == n,
    }


def full_report(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_binary = (scores >= threshold).astype(int)
    out = {"threshold": float(threshold), "anomaly_rate_pred": float(y_binary.mean()),
           "anomaly_rate_true": float(y_true.mean())}
    out["pointwise"] = pointwise(y_true, y_binary)
    out["point_adjust"] = point_adjust(y_true, y_binary)
    out["range_wise"] = range_wise(y_true, y_binary)
    out.update(aucs(y_true, scores))
    out["mean_latency_steps"] = mean_detection_latency(y_true, y_binary)
    return out
