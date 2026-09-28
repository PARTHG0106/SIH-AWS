"""Corrected / imputed value estimation for anomalous observations
(SIH26073 optional output — included).

For each flagged timestamp and channel we propose a corrected value as the
inverse-variance-weighted fusion of three independent estimators:
    1. deep-model reconstruction (learned dynamics)
    2. climatology + damped-persistence blend  (seasonal baseline + last-good)
    3. spatial peer median                     (neighbouring stations / twin
                                                reanalysis series)
The estimator variances are learned from clean training data; the delivered
`confidence` makes the correction self-aware ("how much can we trust it?").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["temperature_c", "pressure_hpa", "relative_humidity_pct"]


def correct_series(raw: pd.Series, resid: pd.Series, clim_val: pd.Series,
                   peer_est: pd.Series | None, ae_est: pd.Series | None,
                   sigma_stats: dict) -> pd.DataFrame:
    """Fuse estimates for one channel across a (possibly anomalous) window.

    sigma_stats: {"ae": float, "clim": float, "peer": float} — clean-data RMSE
    of each estimator for this channel/station (computed at training time).
    """
    n = len(raw)
    out = pd.DataFrame(index=raw.index)

    # climatology + damped persistence of the last trusted residual
    persist = resid.ffill().fillna(0.0).to_numpy()
    decay = np.exp(-np.arange(n) / 12.0)
    clim_est = clim_val.to_numpy(dtype=float) + persist * decay

    ests = [clim_est]
    vars_ = [max(sigma_stats.get("clim", 1.0), 1e-6) ** 2 + persist**2 * 0.05]
    if ae_est is not None:
        ests.append(ae_est.to_numpy(dtype=float))
        vars_.append(max(sigma_stats.get("ae", 1.0), 1e-6) ** 2)
    if peer_est is not None:
        ests.append(peer_est.to_numpy(dtype=float))
        vars_.append(max(sigma_stats.get("peer", 1.0), 1e-6) ** 2)

    w = np.stack([1.0 / v for v in vars_])
    e = np.stack(ests)
    fused = (w * e).sum(axis=0) / w.sum(axis=0)
    sigma = np.sqrt(1.0 / w.sum(axis=0))
    out["corrected"] = fused
    out["confidence"] = np.clip(1.5 / (1.0 + sigma), 0.0, 1.0)
    out["raw"] = raw.to_numpy(dtype=float)
    return out


def summarize_corrections(corr: pd.DataFrame, truth: pd.Series) -> dict:
    m = np.isfinite(truth.to_numpy(dtype=float))
    if not m.any():
        return {"mae": float("nan"), "n": 0}
    err = np.abs(corr["corrected"].to_numpy()[m] - truth.to_numpy(dtype=float)[m])
    return {"mae": float(err.mean()), "n": int(m.sum())}
