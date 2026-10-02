"""Gross range checks on measured T, P, RH; derived dew point is context only.

Out-of-range pressure/temperature/humidity are quality-review proposals, not
ground-truth hardware labels. This layer
complements the forecast-residual detector, which is weak on gross magnitude faults
(clipping / scale_error) that a physical bound catches immediately. Uses only the three
SIH parameters; no external or reanalysis data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from awsad.minute_detection import CHANNELS

# Generous physical envelopes (WMO-style gross-error bounds), not station climatology.
BOUNDS = {"temperature_c": (-90.0, 60.0), "pressure_hpa": (400.0, 1100.0),
          "relative_humidity_pct": (0.0, 100.0)}


def _dewpoint_c(t: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Magnus dew point from air temperature and relative humidity."""
    rh = np.clip(rh, 1e-3, 100.0)
    gamma = np.log(rh / 100.0) + (17.625 * t) / (243.04 + t)
    return 243.04 * gamma / (17.625 - gamma)


def physical_flags(frame: pd.DataFrame) -> dict:
    """Per-row physical-consistency flags + human reasons, using only measured T/P/RH."""
    n = len(frame)
    per_channel, any_flag, reasons = {}, np.zeros(n, bool), np.array([""] * n, dtype=object)

    def _add(mask, tag):
        nonlocal reasons
        reasons[mask] = [(s + "|" if s else "") + tag for s in reasons[mask]]

    for channel, (lo, hi) in BOUNDS.items():
        v = frame[channel].to_numpy(float, na_value=np.nan)
        bad = np.isfinite(v) & ((v < lo) | (v > hi))
        per_channel[channel] = bad
        any_flag |= bad
        _add(bad, f"{channel}:range")
    # Td derived from the same T/RH supplies no independent consistency evidence.
    # Keep it as a derived display feature, never a swap/root-cause detector.
    t = frame["temperature_c"].to_numpy(float, na_value=np.nan)
    rh = frame["relative_humidity_pct"].to_numpy(float, na_value=np.nan)
    ok = np.isfinite(t) & np.isfinite(rh) & (rh > 0)
    dew = np.full(n, np.nan)
    dew[ok] = _dewpoint_c(t[ok], rh[ok])
    inconsistent = np.zeros(n, dtype=bool)
    per_channel["multivariate"] = inconsistent
    any_flag |= inconsistent
    _add(inconsistent, "dewpoint_gt_temperature")
    return {"per_channel": per_channel, "any": any_flag, "reasons": reasons, "dewpoint_c": dew}
