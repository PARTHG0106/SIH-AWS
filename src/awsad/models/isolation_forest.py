"""Isolation Forest branch of the ensemble (windowed-feature detector)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(self, n_estimators: int = 400, max_samples: int | str = 65536,
                 contamination: float = 0.03, random_state: int = 42,
                 n_jobs: int = -1):
        self.model = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples,
            contamination=contamination, random_state=random_state,
            n_jobs=n_jobs, bootstrap=False)
        self.feature_cols: list[str] = []
        self.score_norm: float = 1.0     # train-set q99, for scaling to ~[0,1]

    def fit(self, X: pd.DataFrame, feature_cols: list[str] | None = None) -> "IsolationForestDetector":
        self.feature_cols = list(feature_cols or X.columns)
        Xv = X[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
        Xv[~np.isfinite(Xv)] = 0.0
        self.model.fit(Xv)
        s = -self.model.decision_function(Xv)
        self.score_norm = float(np.quantile(s, 0.99) - np.quantile(s, 0.5)) or 1.0
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, dtype=float)
        Xv = X.reindex(columns=self.feature_cols).to_numpy(dtype=np.float32, copy=True)
        Xv[~np.isfinite(Xv)] = 0.0
        raw = -self.model.decision_function(Xv)
        return raw / self.score_norm

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols,
                     "score_norm": self.score_norm}, path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        blob = joblib.load(path)
        obj = cls()
        obj.model = blob["model"]
        obj.feature_cols = blob["feature_cols"]
        obj.score_norm = blob["score_norm"]
        return obj
