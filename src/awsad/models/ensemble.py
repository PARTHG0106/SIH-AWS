"""Weighted score ensemble with per-station normalisation + threshold state.

Scores from each component (qc / zscore / iforest / lstm_ae / ...) are
normalised per station-group using TRAIN-set statistics, combined with
    validation-calibrated weights, and thresholded per station. Supervised
    thresholds are stored only for stations used during validation calibration;
    train-only POT thresholds are used for held-out and never-before-seen
    stations.
"""
from __future__ import annotations

import json
import numpy as np


def normalize(scores: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    ref = scores if ref is None else ref
    med = np.nanmedian(ref)
    hi = np.nanquantile(ref, 0.99)
    return (scores - med) / max(hi - med, 1e-9)


def _norm_with(stats: tuple[float, float], s: np.ndarray) -> np.ndarray:
    med, hi = float(stats[0]), float(stats[1])
    return (s - med) / max(hi - med, 1e-9)


class WeightedEnsemble:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {}
        self.norms_: dict[str, tuple[float, float]] = {}            # pooled
        self.per_group_norms: dict[str, dict[str, tuple[float, float]]] = {}
        self.per_group_thresholds: dict[str, float] = {}
        self.per_group_fallback_thresholds: dict[str, float] = {}
        self.pooled_threshold: float = 1.0
        self.fallback_threshold: float = 1.0

    # ------------------------------------------------------------- training API
    def fit_norms(self, train_scores: dict[str, np.ndarray]) -> "WeightedEnsemble":
        for k, s in train_scores.items():
            self.norms_[k] = (float(np.nanmedian(s)),
                              float(np.nanquantile(s, 0.99)))
        return self

    def _norm(self, name: str, s: np.ndarray) -> np.ndarray:
        if name not in self.norms_:
            return s
        return _norm_with(self.norms_[name], s)

    def score(self, scores: dict[str, np.ndarray],
              norms: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
        norms = norms or self.norms_
        out = None
        wsum = 0.0
        for k, w in self.weights.items():
            if k in scores and w > 0:
                term = np.nan_to_num(_norm_with(norms[k], scores[k]), nan=0.0) if k in norms \
                    else np.nan_to_num(scores[k], nan=0.0)
                out = term * w if out is None else out + term * w
                wsum += w
        if out is None:
            return np.zeros(0)
        return out / max(wsum, 1e-9)

    def score_group(self, group: str, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Score using this group's norms; pooled fallback for unseen stations."""
        return self.score(scores, self.per_group_norms.get(group) or None)

    def threshold_for(self, group: str | None = None) -> float:
        if group and group in self.per_group_thresholds:
            return self.per_group_thresholds[group]
        if group and group in self.per_group_fallback_thresholds:
            return self.per_group_fallback_thresholds[group]
        return self.fallback_threshold

    # ------------------------------------------------------------- persistence
    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "weights": self.weights,
            "norms": {k: list(v) for k, v in self.norms_.items()},
            "per_group_norms": {g: {k: list(v) for k, v in nv.items()}
                                for g, nv in self.per_group_norms.items()},
            "per_group_thresholds": self.per_group_thresholds,
            "per_group_fallback_thresholds": self.per_group_fallback_thresholds,
            "pooled_threshold": self.pooled_threshold,
            "fallback_threshold": self.fallback_threshold,
        }
        if extra:
            payload.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "WeightedEnsemble":
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        obj = cls(blob.get("weights", {}))
        obj.norms_ = {k: tuple(v) for k, v in blob.get("norms", {}).items()}
        obj.per_group_norms = {g: {k: tuple(v) for k, v in nv.items()}
                               for g, nv in blob.get("per_group_norms", {}).items()}
        obj.per_group_thresholds = {g: float(v) for g, v in
                                    blob.get("per_group_thresholds", {}).items()}
        obj.per_group_fallback_thresholds = {g: float(v) for g, v in
                                             blob.get("per_group_fallback_thresholds", {}).items()}
        obj.pooled_threshold = float(blob.get("pooled_threshold", 1.0))
        obj.fallback_threshold = float(blob.get("fallback_threshold",
                                                obj.pooled_threshold))
        return obj
