"""Threshold selection: quantile fallback + POT (Peaks-Over-Threshold, EVT).

POT (Siffer et al. 2017) fits a Generalised Pareto Distribution to the
exceedances above a high initial quantile and picks the threshold whose
estimated training-tail probability is `risk`. This is not a measured
false-positive rate when the training observations have unknown fault status.
"""
from __future__ import annotations

import numpy as np


def threshold_above(value: float, dtype=np.float64) -> float:
    """Next representable score, including when serving compares float32 data."""
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        dtype = np.dtype(np.float64)
    return float(np.nextafter(np.asarray(value, dtype=dtype),
                              np.asarray(np.inf, dtype=dtype)))


def quantile_threshold(scores: np.ndarray, q: float = 0.995) -> float:
    finite = np.asarray(scores)[np.isfinite(scores)]
    if not len(finite):
        raise ValueError("cannot estimate a threshold without finite training scores")
    threshold = float(np.quantile(finite, q))
    # With the deployed >= comparison a collapsed distribution would flag
    # every observation. Constant reference scores support no such decision.
    if finite.min() == finite.max():
        threshold = threshold_above(threshold, finite.dtype)
    return threshold


def pot_threshold(train_scores: np.ndarray, q_init: float = 0.98,
                  risk: float = 1e-3) -> float:
    s = np.sort(train_scores[np.isfinite(train_scores)])
    if len(s) < 1000:
        return quantile_threshold(s, 0.995)
    t0 = np.quantile(s, q_init)
    exceed = s[s > t0] - t0
    if len(exceed) < 50 or exceed.max() <= 0:
        return quantile_threshold(s, 0.995)
    try:
        from scipy.stats import genpareto
        shape, _, scale = genpareto.fit(exceed, floc=0)
        n, nt = len(s), len(exceed)
        # Inverse GPD survival at target risk level
        with np.errstate(over="ignore", invalid="ignore"):
            if abs(shape) < 1e-9:
                thresh = t0 + scale * np.log(nt / (n * risk))
            else:
                thresh = t0 + scale / shape * ((n * risk / nt) ** (-shape) - 1)
        if not np.isfinite(thresh) or thresh <= t0:
            return quantile_threshold(s, 0.995)
        return float(thresh)
    except Exception:
        return quantile_threshold(s, 0.995)
