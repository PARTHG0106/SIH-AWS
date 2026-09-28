"""Review-gated real-event evaluation; unknown intervals are never negatives.

Inputs are long-form per-channel detector decisions on original observations.
Review intervals are UTC [start_utc, end_utc). A confirmed event is one reviewed
event-channel interval, not every flagged point and not a hardware diagnosis.
Candidate extraction and background sampling create review proposals only.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np
import pandas as pd


REVIEW_COLUMNS = [
    "review_id", "event_id", "group", "channel", "start_utc", "end_utc",
    "review_status", "onset_earliest_utc", "onset_latest_utc", "reviewer",
    "reviewed_at_utc", "evidence_basis", "evidence_refs", "review_notes",
]
_EVIDENCE_BASES = {"maintenance_record", "calibration_record", "independent_measurement", "operator_log"}
_CANDIDATE_COLUMNS = [
    "event_id", "group", "station_id", "source", "channel", "split", "start", "end",
    "observation_count", "max_score", "reason_codes", "first_observation_id",
    "last_observation_id", "review_status", "candidate_kind",
]


def _cadence(seconds: float) -> pd.Timedelta:
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("expected_cadence_seconds must be positive and finite")
    return pd.Timedelta(seconds=seconds)


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None:
        raise ValueError(f"{name} requires an explicit UTC offset")
    return ts.tz_convert("UTC")


def _decisions(alerts: pd.DataFrame) -> pd.DataFrame:
    required = {"group", "timestamp", "channel", "observation_id", "is_alert"}
    if not required <= set(alerts.columns):
        raise ValueError(f"Decisions missing columns: {sorted(required - set(alerts.columns))}")
    out = alerts.copy()
    for key in ("group", "channel", "observation_id"):
        if out[key].isna().any() or out[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"{key} must be present for every observation")
    if not out["group"].astype(str).str.contains("|", regex=False).all():
        raise ValueError("group must retain station|source granularity")
    # Nullable booleans preserve rows the detector could not evaluate. Do not
    # accept strings: bool('False') would silently manufacture an alarm.
    if not out["is_alert"].dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("is_alert must contain booleans or missing decisions")
    out["is_alert"] = out["is_alert"].astype("boolean")
    if len(out):
        # Vectorized for millions of minute records; datetimes must include TZ.
        if isinstance(out["timestamp"].dtype, pd.DatetimeTZDtype):
            out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")
        else:
            text = out["timestamp"].astype(str)
            if not text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
                raise ValueError("timestamp requires an explicit UTC offset")
            out["timestamp"] = pd.to_datetime(text, utc=True, errors="raise", format="mixed")
        if out["timestamp"].isna().any():
            raise ValueError("timestamp cannot be missing")
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if out.duplicated(["group", "channel", "timestamp"]).any():
        raise ValueError("Duplicate group/channel/timestamp decisions")
    if out.duplicated(["observation_id", "channel"]).any():
        raise ValueError("Duplicate observation_id/channel decisions")
    if "split" not in out:
        out["split"] = "unspecified"
    if out["split"].isna().any():
        raise ValueError("split cannot be missing when supplied")
    return out.sort_values(["group", "channel", "timestamp"], kind="stable").reset_index(drop=True)


def _runs(frame: pd.DataFrame, cadence: pd.Timedelta, wanted: bool):
    for _, group in frame.groupby(["group", "channel", "split"], sort=True, observed=True):
        selected = group[group["is_alert"].eq(wanted).fillna(False)]
        if selected.empty:
            continue
        run_id = selected["timestamp"].diff().ne(cadence).cumsum()
        for _, run in selected.groupby(run_id, sort=False):
            yield run


def _candidate(run: pd.DataFrame, kind: str) -> dict[str, Any]:
    first, last = run.iloc[0], run.iloc[-1]
    identity = [kind, first["group"], first["channel"], first["split"],
                first["observation_id"], last["observation_id"]]
    event_id = "candidate-" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
    scores = pd.to_numeric(run.get("score", pd.Series(dtype=float)), errors="coerce")
    scores = scores[np.isfinite(scores)]
    reasons: set[str] = set()
    for values in run.get("reason_codes", []):
        if isinstance(values, str):
            try:
                values = json.loads(values)
            except json.JSONDecodeError:
                values = [values]
        if isinstance(values, str):
            values = [values]
        if isinstance(values, (list, tuple, np.ndarray)):
            reasons.update(str(value) for value in values)
    station, source = str(first["group"]).split("|", 1)
    return {
        "event_id": event_id, "group": first["group"], "station_id": station, "source": source,
        "channel": first["channel"], "split": first["split"],
        "start": first["timestamp"], "end": last["timestamp"], "observation_count": len(run),
        "max_score": float(scores.max()) if len(scores) else None,
        "reason_codes": json.dumps(sorted(reasons)),
        "first_observation_id": first["observation_id"], "last_observation_id": last["observation_id"],
        "review_status": "unknown", "candidate_kind": kind,
    }


def extract_candidate_events(alerts: pd.DataFrame, *, expected_cadence_seconds: float = 60) -> pd.DataFrame:
    """Merge adjacent alert decisions; gaps, missing decisions and splits break runs."""
    cadence = _cadence(expected_cadence_seconds)
    decisions = _decisions(alerts)
    return pd.DataFrame([_candidate(run, "detector_alert") for run in _runs(decisions, cadence, True)],
                        columns=_CANDIDATE_COLUMNS)


def sample_background_candidates(
    alerts: pd.DataFrame, *, max_per_group: int = 3, window_minutes: float = 60,
    expected_cadence_seconds: float = 60,
) -> pd.DataFrame:
    """Propose non-alert intervals for independent review, NEVER normal labels.

    Non-overlapping windows use actual consecutive evaluated rows. At most
    max_per_group windows per group/channel/split are selected at evenly spaced
    indices. This purposeful sample is not a random population error estimate.
    """
    cadence = _cadence(expected_cadence_seconds)
    if not isinstance(max_per_group, int) or max_per_group < 0:
        raise ValueError("max_per_group must be a nonnegative integer")
    if not np.isfinite(window_minutes) or window_minutes <= 0:
        raise ValueError("window_minutes must be positive and finite")
    rows_per_window = max(1, int(window_minutes * 60 / expected_cadence_seconds))
    decisions = _decisions(alerts)
    windows: dict[tuple, list[pd.DataFrame]] = {}
    for run in _runs(decisions, cadence, False):
        key = tuple(run.iloc[0][["group", "channel", "split"]])
        for start in range(0, len(run), rows_per_window):
            windows.setdefault(key, []).append(run.iloc[start:start + rows_per_window])
    records = []
    for alternatives in windows.values():
        positions = np.linspace(0, len(alternatives) - 1, min(max_per_group, len(alternatives)), dtype=int)
        records.extend(_candidate(alternatives[i], "non_alert_review_sample") for i in positions)
    return pd.DataFrame(records, columns=_CANDIDATE_COLUMNS)


def review_template(candidates: pd.DataFrame | None = None, *, expected_cadence_seconds: float = 60) -> pd.DataFrame:
    """Create an editable review queue; its observations never become truth here."""
    cadence = _cadence(expected_cadence_seconds)
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    required = {"event_id", "group", "channel", "start", "end"}
    if not required <= set(candidates):
        raise ValueError(f"Candidates missing columns: {sorted(required - set(candidates))}")
    records = []
    for row in candidates.to_dict("records"):
        start = _timestamp(row["start"], "start")
        end = _timestamp(row["end"], "end")
        if end < start:
            raise ValueError("Candidate end precedes start")
        records.append({
            "review_id": "review-" + str(row["event_id"]), "event_id": row["event_id"],
            "group": row["group"], "channel": row["channel"],
            "start_utc": start.isoformat(), "end_utc": (end + cadence).isoformat(),
            "review_status": "unknown", "onset_earliest_utc": None, "onset_latest_utc": None,
            "reviewer": None, "reviewed_at_utc": None, "evidence_basis": None,
            "evidence_refs": "[]", "review_notes": "Review proposal only; absence of an alert or QC flag is not normal truth.",
        })
    return pd.DataFrame(records, columns=REVIEW_COLUMNS)


def validate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Reject contradictory, untraceable or QC/model-derived benchmark truth.

    This checks declarations and interval integrity; a reviewer still must
    inspect the cited bytes and establish what they support. It does not turn
    a citation into independently verified truth automatically.
    """
    if reviews.empty:
        return review_template()
    required = set(REVIEW_COLUMNS) - {"review_notes"}
    if not required <= set(reviews):
        raise ValueError(f"Reviews missing columns: {sorted(required - set(reviews))}")
    out = reviews.copy()
    for column in ("start_utc", "end_utc", "onset_earliest_utc", "onset_latest_utc"):
        out[column] = out[column].astype(object)
    if out["review_id"].isna().any() or out["review_id"].duplicated().any():
        raise ValueError("Every review_id must be present and unique")
    allowed = {"unknown", "confirmed_event", "reviewed_background"}
    if not out["review_status"].isin(allowed).all():
        raise ValueError("Unknown review_status; use unknown, confirmed_event or reviewed_background")
    for i, row in out.iterrows():
        for key in ("group", "channel"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Review {key} must be present")
        if "|" not in row["group"]:
            raise ValueError("Review group must retain station|source granularity")
        start, end = _timestamp(row["start_utc"], "start_utc"), _timestamp(row["end_utc"], "end_utc")
        if end <= start:
            raise ValueError("Review intervals must have start_utc < end_utc")
        out.at[i, "start_utc"], out.at[i, "end_utc"] = start, end
        if row["review_status"] == "unknown":
            continue
        if not isinstance(row["reviewer"], str) or not row["reviewer"].strip():
            raise ValueError("Accepted reviews require a named reviewer")
        _timestamp(row["reviewed_at_utc"], "reviewed_at_utc")
        if row["evidence_basis"] not in _EVIDENCE_BASES:
            raise ValueError("Accepted reviews require independent evidence; QC and model outputs are not fault truth")
        refs = row["evidence_refs"]
        if isinstance(refs, str):
            refs = json.loads(refs)
        if not isinstance(refs, list) or not refs:
            raise ValueError("Accepted reviews require evidence_refs with URL and immutable SHA-256")
        for ref in refs:
            if (not isinstance(ref, dict) or not re.fullmatch(r"[a-fA-F0-9]{64}", str(ref.get("sha256", "")))
                    or not str(ref.get("url", "")).startswith(("https://", "http://"))):
                raise ValueError("Every evidence reference requires a source URL and SHA-256")
        if row["review_status"] == "confirmed_event":
            if not isinstance(row["event_id"], str) or not row["event_id"].strip():
                raise ValueError("Confirmed events require event_id")
            early = _timestamp(row["onset_earliest_utc"], "onset_earliest_utc")
            late = _timestamp(row["onset_latest_utc"], "onset_latest_utc")
            if early > late or late > start:
                raise ValueError("Onset bounds must satisfy earliest <= latest <= reviewed event start")
            out.at[i, "onset_earliest_utc"], out.at[i, "onset_latest_utc"] = early, late
    accepted = out[out["review_status"] != "unknown"]
    for _, group in accepted.groupby(["group", "channel"], sort=False):
        ordered = group.sort_values("start_utc")
        ends = ordered["end_utc"].tolist()
        if any(start < end for start, end in zip(ordered["start_utc"].tolist()[1:], ends[:-1])):
            raise ValueError("Accepted review intervals overlap within group/channel")
    events = accepted[accepted["review_status"] == "confirmed_event"]
    if events.duplicated(["group", "channel", "event_id"]).any():
        raise ValueError("One event_id must not be counted twice within group/channel")
    return out


def evaluate_reviewed_events(
    alerts: pd.DataFrame, reviews: pd.DataFrame, *, expected_cadence_seconds: float = 60,
) -> dict[str, Any]:
    """Measure only explicitly reviewed event-channel/background intervals.

    All observed decisions (including False and missing) must be supplied; an
    alerts-only table cannot establish exposure. Event recall includes reviewed
    events with no scored observations as misses and reports their number.
    Detection delay is bounded by documented onset uncertainty. No precision or
    population accuracy is inferred from an alert-selected review sample.
    """
    cadence = _cadence(expected_cadence_seconds)
    decisions = _decisions(alerts)
    reviewed = validate_reviews(reviews)
    accepted = reviewed[reviewed["review_status"] != "unknown"]
    lookup = {key: group for key, group in decisions.groupby(["group", "channel"], sort=False)}
    used = np.zeros(len(decisions), dtype=bool)
    event_results = []
    background_results = []
    for row in accepted.to_dict("records"):
        group = lookup.get((row["group"], row["channel"]), decisions.iloc[:0])
        part = group[(group["timestamp"] >= row["start_utc"]) & (group["timestamp"] < row["end_utc"])]
        used[part.index] = True
        evaluated = part[part["is_alert"].notna()]
        hits = evaluated[evaluated["is_alert"].fillna(False)]
        if row["review_status"] == "confirmed_event":
            first = hits.iloc[0] if len(hits) else None
            event_results.append({
                "review_id": row["review_id"], "event_id": row["event_id"], "group": row["group"],
                "channel": row["channel"], "observed_rows": len(part), "scored_rows": len(evaluated),
                "detected": first is not None,
                "first_alert_utc": first["timestamp"].isoformat() if first is not None else None,
                "first_alert_observation_id": first["observation_id"] if first is not None else None,
                "delay_seconds_min": float((first["timestamp"] - row["onset_latest_utc"]).total_seconds()) if first is not None else None,
                "delay_seconds_max": float((first["timestamp"] - row["onset_earliest_utc"]).total_seconds()) if first is not None else None,
            })
        else:
            # Continuous monitored time is only adjacent scored timestamp pairs
            # with exactly the provider cadence. Gaps and splits add no exposure.
            consecutive = evaluated["timestamp"].diff().eq(cadence) & evaluated["split"].eq(evaluated["split"].shift())
            exposure_seconds = float(consecutive.sum() * expected_cadence_seconds)
            episodes = sum(1 for _ in _runs(part, cadence, True))
            background_results.append({
                "review_id": row["review_id"], "group": row["group"], "channel": row["channel"],
                "reviewed_interval_seconds": float((row["end_utc"] - row["start_utc"]).total_seconds()),
                "observed_rows": len(part), "scored_rows": len(evaluated), "false_alert_rows": len(hits),
                "false_alert_episodes": episodes, "contiguous_scored_seconds": exposure_seconds,
            })
    detected = [event for event in event_results if event["detected"]]
    scored_background = sum(row["scored_rows"] for row in background_results)
    false_rows = sum(row["false_alert_rows"] for row in background_results)
    false_episodes = sum(row["false_alert_episodes"] for row in background_results)
    exposure = sum(row["contiguous_scored_seconds"] for row in background_results)
    return {
        "status": "reviewed_intervals_only" if len(accepted) else "unverified_no_reviewed_truth",
        "unit": "event_channel_interval", "input_observation_rows": len(decisions),
        "reviewed_observation_rows": int(used.sum()), "unknown_observation_rows": int((~used).sum()),
        "unknown_alert_rows": int(decisions.loc[~used, "is_alert"].fillna(False).sum()),
        "unknown_review_intervals": int((reviewed["review_status"] == "unknown").sum()),
        "confirmed_event_count": len(event_results), "detected_event_count": len(detected),
        "events_without_scored_observations": sum(event["scored_rows"] == 0 for event in event_results),
        "event_recall": len(detected) / len(event_results) if event_results else None,
        "mean_detection_delay_seconds_min": float(np.mean([event["delay_seconds_min"] for event in detected])) if detected else None,
        "mean_detection_delay_seconds_max": float(np.mean([event["delay_seconds_max"] for event in detected])) if detected else None,
        "reviewed_background_intervals": len(background_results), "background_scored_rows": scored_background,
        "false_alert_rows": false_rows if background_results else None,
        "false_alert_fraction": false_rows / scored_background if scored_background else None,
        "false_alert_episodes": false_episodes if background_results else None,
        "background_contiguous_scored_seconds": exposure,
        "false_alert_episodes_per_24_contiguous_scored_hours": false_episodes * 86400 / exposure if exposure else None,
        "events": event_results, "background": background_results,
        "scope_note": "Metrics apply only to independently reviewed intervals. Unknown rows are excluded; selected review samples do not establish population accuracy.",
    }
