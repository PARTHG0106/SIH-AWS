"""Sensor health index & predictive maintenance (SIH26073 objectives).

Health(t) is a 0..100 index from the EWMA of recent ensemble scores relative
to the station's alert threshold; its trend gives a degradation slope used to
forecast "maintenance due in N days".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def health_index(scores: np.ndarray, threshold: float,
                 halflife_h: int = 72) -> np.ndarray:
    s = pd.Series(scores)
    e = s.ewm(halflife=halflife, adjust=False).mean().to_numpy()
    return 100.0 * np.exp(-np.clip(e / max(threshold, 1e-9) - 1.0, 0, None))


def predict_maintenance(daily_health: pd.Series) -> dict:
    """Linear trend on health; returns days-until-threshold forecast."""
    y = daily_health.dropna().to_numpy(dtype=float)
    if len(y) < 10:
        return {"days_to_maintenance": None, "slope_per_day": 0.0,
                "status": "insufficient_history"}
    t = np.arange(len(y))
    slope, intercept = np.polyfit(t, y, 1)
    status = "ok"
    if slope < -0.05:
        days = (y[-1] - 60.0) / abs(slope)
        status = "degrading" if days > 30 else "maintenance_soon"
        return {"days_to_maintenance": round(float(days), 1),
                "slope_per_day": round(float(slope), 4), "status": status}
    return {"days_to_maintenance": None, "slope_per_day": round(float(slope), 4),
            "status": "ok"}


def station_health_report(scores: np.ndarray, timestamps: pd.Series,
                          threshold: float) -> dict:
    hi = health_index(scores, threshold)
    s = pd.Series(hi, index=pd.to_datetime(timestamps).values)
    daily = s.resample("1D").mean().dropna()
    maint = predict_maintenance(daily)
    return {"health_now": round(float(hi[-1]), 1),
            "health_7d_mean": round(float(s.last("7D").mean()), 1),
            **maint}
