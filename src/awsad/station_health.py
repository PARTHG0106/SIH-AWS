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
_INSPECTION_ACTIONS = {
    "flatline_minutes": "Check sensor resolution, stale logger values, telemetry refresh and the source report.",
    "abrupt_change": "Inspect the abrupt transition and subsequent recovery; compare source reports, sensor wiring and logger diagnostics.",
    "forecast_residual": "Compare adjacent original reports and logger diagnostics; inspect the sensor if the discrepancy persists.",
    "hour_residual": "Review weather evolution and calibration records before scheduling a sensor inspection.",
    "sustained_deviation": "Review calibration history and the sustained weather change; request an independent reference check if the mismatch persists.",
}


def _maintenance_advisory(channel: dict, rows: int, window_minutes: float | None) -> dict:
    """Transparent service triage policy, not a learned lifetime forecast."""
    scored = channel["scored_rows"]
    coverage = scored / rows if rows else None
    adequate = scored >= 30 and window_minutes is not None and window_minutes >= 60
    early, recent = channel["early_candidate_rate"], channel["recent_candidate_rate"]
    persistent = early is not None and recent is not None and min(early, recent) > .02
    inspection = _INSPECTION_ACTIONS.get(channel["dominant_reason"],
                    "Inspect the original reports and station diagnostics before selecting a hardware intervention.")
    if not adequate:
        state = "insufficient_history"
        priority = "review_candidates" if channel["alerts"] else "collect_more_history"
        action = (inspection + " " if channel["alerts"] else "") + "Collect at least 30 scored readings across one hour before assessing a temporal maintenance trend."
        if rows and scored == 0:
            priority = "check_feed_now"
            action = "Check missing reports, transport and logger status; no scored evidence is available for sensor-health triage."
    elif coverage < .5 or (channel["missing_fraction"] is not None and channel["missing_fraction"] > .25):
        state, priority = "insufficient_coverage", "check_feed_now"
        action = "Investigate data availability and restore original observations; the scored subset cannot establish overall hardware condition. " + inspection
    elif channel["trend"] == "worsening":
        state, priority, action = "rising_candidate_activity", "review_now", inspection
    elif persistent or channel["state"] in ("degraded", "critical"):
        state = "persistent_candidate_activity" if persistent else "elevated_candidate_activity"
        priority = "review_now" if channel["state"] == "critical" else "next_service"
        action = inspection
    elif channel["alerts"]:
        state, priority, action = "isolated_candidate_activity", "next_service", inspection
    else:
        state, priority = "low_candidate_activity", "routine_observation"
        action = "Continue recording original observations and follow the station's existing inspection schedule."
    return {"state": state, "priority": priority, "action": action,
            "history_sufficient_for_policy": adequate,
            "evidence": {"scored_rows": scored, "received_rows": rows, "scored_fraction": coverage,
                         "window_minutes": window_minutes, "candidate_rate": channel["candidate_rate"],
                         "early_candidate_rate": early, "recent_candidate_rate": recent,
                         "dominant_signal": channel["dominant_reason"]},
            "policy": {"minimum_scored_rows": 30, "minimum_window_minutes": 60,
                       "minimum_scored_fraction": .5, "maximum_missing_fraction": .25},
            "semantics": "Service priority is a review policy based on candidate activity and coverage, not confirmed degradation or a failure-time prediction."}


def _state(rate: float | None) -> str:
    if rate is None:
        return "unknown"
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


def station_health(frame: pd.DataFrame, *, expected_cadence_minutes: int | None = None) -> dict:
    """Use scored alerts as denominator; compare elapsed-time halves for trend.

    Missing fractions describe received rows. Unreported slots are separate and
    require explicit cadence. ``healthy`` means low candidate activity only.
    """
    if expected_cadence_minutes is not None and (
            isinstance(expected_cadence_minutes, bool) or expected_cadence_minutes < 1):
        raise ValueError("expected cadence must be a positive number of minutes")
    n = len(frame)
    times = (pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
             if "timestamp" in frame else pd.Series(pd.NaT, index=frame.index))
    first = times.min() if times.notna().any() else None
    last = times.max() if times.notna().any() else None
    midpoint = first + (last - first) / 2 if first is not None and first < last else None
    window_minutes = (last - first).total_seconds() / 60 if first is not None else None
    early_mask = times.lt(midpoint).to_numpy() if midpoint is not None else np.zeros(n, bool)
    late_mask = (times.ge(midpoint) & times.notna()).to_numpy() if midpoint is not None else np.zeros(n, bool)
    channels = {}
    known_states = []
    for c in CHANNELS:
        alert_col = f"{c}__alert"
        flags = frame[alert_col] if alert_col in frame else pd.Series(pd.NA, index=frame.index)
        scored = flags.notna().to_numpy()
        alerts = flags.fillna(False).to_numpy(dtype=bool) & scored
        scored_rows = int(scored.sum())
        rate = float(alerts.sum() / scored_rows) if scored_rows else None

        def candidate_rate(mask):
            denominator = int((scored & mask).sum())
            return float((alerts & mask).sum() / denominator) if denominator else None

        early, late = candidate_rate(early_mask), candidate_rate(late_mask)
        trend = "unknown"
        if early is not None and late is not None:
            trend = "worsening" if late > early * 1.5 + 0.005 else ("improving" if early > late * 1.5 + 0.005 else "stable")
        values = (pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                  if c in frame else np.full(n, np.nan))
        missing = int((~np.isfinite(values)).sum())
        reason_col = f"{c}__reason_codes"
        state = _state(rate)
        if state != "unknown":
            known_states.append(state)
        channels[c] = {"candidate_rate": rate, "state": state, "trend": trend,
                       "alerts": int(alerts.sum()), "scored_rows": scored_rows,
                       "unscored_rows": n - scored_rows, "missing_rows": missing,
                       "missing_fraction": missing / n if n else None,
                       "early_candidate_rate": early, "recent_candidate_rate": late,
                       "early_scored_rows": int((scored & early_mask).sum()),
                       "recent_scored_rows": int((scored & late_mask).sum()),
                       "dominant_reason": _dominant_reason(frame.loc[alerts, reason_col]) if reason_col in frame else None}
        channels[c]["maintenance_advisory"] = _maintenance_advisory(channels[c], n, window_minutes)
    unreported = None
    if expected_cadence_minutes is not None and times.notna().any():
        cadence = pd.Timedelta(minutes=expected_cadence_minutes)
        deltas = times.dropna().drop_duplicates().sort_values().diff().dropna()
        unreported = int(sum(max(0, int(delta // cadence) - 1) for delta in deltas))
    priorities = {"routine_observation": 0, "collect_more_history": 1, "next_service": 2,
                  "review_candidates": 3, "check_feed_now": 4, "review_now": 4}
    selected_channel = max(channels, key=lambda c: priorities[channels[c]["maintenance_advisory"]["priority"]])
    advisory = {**channels[selected_channel]["maintenance_advisory"], "channel": selected_channel}
    return {"overall_state": max(known_states, key=STATES.index) if known_states else "unknown",
            "rows": int(n), "channels": channels,
            "maintenance_advisory": advisory,
            "first_timestamp": first.isoformat() if first is not None else None,
            "last_timestamp": last.isoformat() if last is not None else None,
            "trend_midpoint": midpoint.isoformat() if midpoint is not None else None,
            "expected_cadence_minutes": expected_cadence_minutes, "unreported_slots": unreported,
            "hardware_fault_status": "unknown", "remaining_useful_life": None,
            "semantics": "Candidate alerts / scored rows in this received window. States are review triage bands; healthy means low candidate activity, not verified hardware health. Trend compares elapsed-time halves; no lifetime prediction."}
