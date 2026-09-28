"""Rule-based fault-TYPE attribution for detected anomalous segments.

Once the ensemble flags an anomalous time range, we explain it: the operator
needs "temperature sensor stuck" not just "anomaly=0.97". Deterministic
residual-signature rules classify the dominant fault pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CORE = ["temperature_c", "pressure_hpa", "relative_humidity_pct",
        "dewpoint_c", "td_spread", "vpd_hpa"]


def _channel_of_interest(df: pd.DataFrame, residuals: pd.DataFrame) -> str:
    """Channel with the largest robust-scaled absolute residual in-window."""
    best, best_v = CORE[0], -np.inf
    for c in CORE:
        if c not in residuals or c not in df:
            continue
        v = residuals[c].abs().max()
        if np.isfinite(v) and v > best_v:
            best, best_v = c, float(v)
    return best


def classify_event(clean_ctx: pd.DataFrame, window: pd.DataFrame,
                   channel_hint: str | None = None,
                   qc_window: pd.DataFrame | None = None) -> dict:
    # QC hard evidence wins outright when present
    if qc_window is not None and len(qc_window):
        for c in qc_window.columns:
            if c.startswith("flatline_") and qc_window[c].any():
                return {"fault_type": "stuck",
                        "channel": c.replace("flatline_", ""), "confidence": 0.99}
        for c in qc_window.columns:
            if c.startswith("range_") and qc_window[c].any():
                return {"fault_type": "spike",
                        "channel": c.replace("range_", ""), "confidence": 0.95,
                        "detail": "out of physical range"}
    """Classify one anomalous window.

    clean_ctx: preceding clean reference window (same length or longer).
    window:    the anomalous rows (raw corrupted values, schema columns).
    """
    res = {}
    for c in CORE:
        if c in window and c in clean_ctx:
            ref = clean_ctx[c].to_numpy(dtype=float)
            if not np.isfinite(ref).any():
                continue
            res[c] = window[c].to_numpy(dtype=float) - np.nanmedian(ref)
    ch = channel_hint or _channel_of_interest(window, pd.DataFrame(res))
    if ch not in window:
        return {"fault_type": "unknown", "channel": ch, "confidence": 0.0}

    x = window[ch].to_numpy(dtype=float)
    ref = clean_ctx[ch].to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return {"fault_type": "unknown", "channel": ch, "confidence": 0.0}

    nan_frac = float(np.isnan(x).mean())
    x_filled = pd.Series(x).ffill().bfill().to_numpy()
    std_ref = np.nanstd(ref) or 1e-6
    var_ratio = (np.nanstd(x_filled) / std_ref) if std_ref > 0 else 1.0

    if nan_frac > 0.5:
        return {"fault_type": "dropout", "channel": ch, "confidence": 0.95}
    if n < 3:
        return {"fault_type": "spike", "channel": ch, "confidence": 0.5}
    if np.nanmax(np.abs(np.diff(x_filled))) < 1e-8 and n >= 6:
        return {"fault_type": "stuck", "channel": ch, "confidence": 0.95}
    if (x_filled == 0).mean() > 0.9:
        return {"fault_type": "dropout", "channel": ch,
                "confidence": 0.9, "detail": "zero-fill"}

    dev = (x_filled - np.nanmedian(ref)) / std_ref
    t = np.arange(n)
    if n >= 6:
        slope = np.polyfit(t, dev, 1)[0]
        if abs(slope) * n > 3 and abs(np.corrcoef(t, dev)[0, 1]) > 0.85:
            return {"fault_type": "drift", "channel": ch,
                    "confidence": min(0.95, 0.5 + abs(slope) / (abs(slope) + 1))}
    if var_ratio > 3.0:
        return {"fault_type": "noise_burst", "channel": ch,
                "confidence": min(0.95, 0.5 + var_ratio / (var_ratio + 5))}
    if var_ratio < 0.4 and n >= 12:
        return {"fault_type": "scale_error", "channel": ch, "confidence": 0.7}
    q99 = np.nanquantile(np.abs(dev), 0.99)
    if abs(np.nanmean(dev)) > 1.5 and var_ratio < 1.5:
        return {"fault_type": "bias", "channel": ch, "confidence": 0.75}
    if np.nanmax(np.abs(dev)) > 4 and n <= 6:
        return {"fault_type": "spike", "channel": ch, "confidence": 0.9}
    if q99 > 3:
        return {"fault_type": "spike", "channel": ch, "confidence": 0.6}
    if (x_filled >= np.nanquantile(ref, 0.98)).mean() > 0.8 and var_ratio < 0.6:
        return {"fault_type": "clipping", "channel": ch, "confidence": 0.7}
    return {"fault_type": "complex", "channel": ch, "confidence": 0.5,
            "detail": "pattern consistent with multiple/compound faults"}
