"""Per-station sensor-health summary from scored real observations.

Health is a rolling CANDIDATE rate (proposals for review), never a confirmed
hardware-fault rate. States are triage bands; a degradation trend compares the
recent half of the window with the earlier half. Reads only the detector's own
scored outputs — no fabricated labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from awsad.minute_detection import CHANNELS

STATES = ("healthy", "watch", "degraded", "critical")
_BANDS = ((0.02, "healthy"), (0.08, "watch"), (0.25, "degraded"))


def _state(rate: float) -> str:
    for edge, name in _BANDS:
        if rate <= edge:
            return name
    return "critical"


def _dominant_reason(series: pd.Series) -> str | None:
    codes: dict[str, int] = {}
    for cell in series.dropna():
        for token in str(cell).split("|"):
            if token:
                codes[token] = codes.get(token, 0) + 1
    return max(codes, key=codes.get) if codes else None


def station_health(frame: pd.DataFrame) -> dict:
    """Summarise per-channel candidate rate, state and trend over the given window."""
    n = len(frame)
    channels = {}
    worst = "healthy"
    for c in CHANNELS:
        alert_col = f"{c}__alert"
        alerts = (frame[alert_col].fillna(False).to_numpy(dtype=bool) if alert_col in frame
                  else np.zeros(n, bool))
        rate = float(alerts.mean()) if n else 0.0
        half = n // 2
        early = float(alerts[:half].mean()) if half else 0.0
        late = float(alerts[half:].mean()) if n - half else 0.0
        trend = "worsening" if late > early * 1.5 + 0.005 else ("improving" if early > late * 1.5 + 0.005 else "stable")
        reason_col = f"{c}__reason_codes"
        state = _state(rate)
        if STATES.index(state) > STATES.index(worst):
            worst = state
        channels[c] = {"candidate_rate": rate, "state": state, "trend": trend,
                       "alerts": int(alerts.sum()),
                       "dominant_reason": _dominant_reason(frame[reason_col]) if reason_col in frame else None}
    return {"overall_state": worst, "rows": int(n), "channels": channels,
            "semantics": "rolling candidate (review-proposal) rate; not a confirmed hardware-fault rate"}
