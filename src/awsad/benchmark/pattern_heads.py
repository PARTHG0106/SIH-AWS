"""Separate change-detection and conditional scenario-type learning heads.

The background target means no injected intervention. This is a model of the
declared software scenarios, not independently labelled hardware condition.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class TwoHeadPatternClassifier:
    def __init__(self, *, seed=26073, max_iter=170, max_leaf_nodes=31):
        self.seed, self.max_iter, self.max_leaf_nodes = seed, max_iter, max_leaf_nodes

    def _model(self):
        return HistGradientBoostingClassifier(max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes, learning_rate=.08, min_samples_leaf=20,
            l2_regularization=2., early_stopping=False, random_state=self.seed)

    def fit(self, X, y, sample_weight=None):
        labels = np.asarray(y)
        changed = labels != "no_injection"
        counts = np.bincount(changed.astype(int), minlength=2)
        if not counts.all():
            raise ValueError("both unmodified backgrounds and modified scenarios are required")
        detection_weight = np.sqrt(len(labels) / counts)[changed.astype(int)]
        self.detector_ = self._model()
        self.detector_.fit(X, changed.astype(int), sample_weight=detection_weight)
        self.typer_ = self._model()
        weights = np.asarray(sample_weight)[changed] if sample_weight is not None else None
        self.typer_.fit(X[changed], labels[changed], sample_weight=weights)
        self.classes_ = np.concatenate((np.array(["no_injection"]), self.typer_.classes_))
        return self

    def predict_proba(self, X):
        change = self.detector_.predict_proba(X)[:, 1]
        conditional = self.typer_.predict_proba(X)
        return np.column_stack((1 - change, change[:, None] * conditional))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
