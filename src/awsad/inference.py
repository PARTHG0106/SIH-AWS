"""Serve the trained supervised stacking meta-learner on new data.

The stacker consumes, per timestamp: the per-group-normalized component detector
scores, the robust-scaled residual/window features (``fcols``), and the QC
hard-flag.  ``stack_meta_features`` fixes the column order so that training and
inference build byte-identical matrices.  :class:`SupervisedScorer` loads
``supervised_meta.joblib`` and turns that matrix into calibrated anomaly
probabilities plus per-station binary flags.

Full network inference (spatial buddy-check needs neighbours + the deep heads)
re-runs the detector stack from the saved artifacts and then applies this
scorer; the streaming fast-path in :mod:`awsad.streaming` remains a reduced,
QC-only detector for per-observation latency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def stack_meta_features(norm_scores: np.ndarray, feats: np.ndarray,
                        hard: np.ndarray) -> np.ndarray:
    """Assemble the meta-learner feature matrix in a FIXED column layout:
    ``[normalized component scores | fcols | qc_hard_flag]``.

    Shared by the training pipeline and inference so the column order the model
    was fit on is exactly reproduced at serving time.
    """
    norm_scores = np.asarray(norm_scores, dtype=np.float32)
    feats = np.asarray(feats, dtype=np.float32)
    hard = np.asarray(hard, dtype=np.float32).reshape(-1, 1)
    if norm_scores.ndim != 2 or feats.ndim != 2:
        raise ValueError("norm_scores and feats must be 2-D")
    if not (len(norm_scores) == len(feats) == len(hard)):
        raise ValueError("row count mismatch across meta-feature blocks")
    return np.hstack([norm_scores, feats, hard])


class SupervisedScorer:
    """Load ``supervised_meta.joblib`` and score prepared meta-feature matrices.

    The matrix must be built with :func:`stack_meta_features` from the SAME
    ``components`` order and ``fcols`` the model was trained with (both are
    stored in the artifact).
    """

    def __init__(self, model, components, fcols, pooled_threshold,
                 per_group_thresholds=None):
        self.model = model
        self.components = list(components)
        self.fcols = list(fcols)
        self.pooled_threshold = float(pooled_threshold)
        self.per_group_thresholds = dict(per_group_thresholds or {})

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedScorer":
        import joblib
        blob = joblib.load(Path(path))
        return cls(blob["model"], blob["components"], blob.get("fcols", []),
                   blob["pooled_threshold"], blob.get("per_group_thresholds", {}))

    def n_features(self) -> int:
        return len(self.components) + len(self.fcols) + 1

    def threshold_for(self, group: str | None = None) -> float:
        return self.per_group_thresholds.get(group, self.pooled_threshold)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features():
            raise ValueError(
                f"expected {self.n_features()} meta-features "
                f"({len(self.components)} scores + {len(self.fcols)} fcols + 1 hard), "
                f"got {X.shape[1]}")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, group: str | None = None):
        """Return (probability, binary flag) using the group's threshold."""
        proba = self.predict_proba(X)
        return proba, (proba >= self.threshold_for(group)).astype(int)
