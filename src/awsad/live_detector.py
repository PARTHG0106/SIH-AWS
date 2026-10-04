"""Bounded causal streaming with frozen native-minute models and thresholds.

Only three observed channels enter predictions. Threshold ratios are not
probabilities; patterns and maintenance actions are investigation suggestions.
Missing values and overdue slots never become observations. State is ephemeral;
source records must remain in their original immutable archive.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Callable, Mapping

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from awsad.minute_detection import CHANNELS, SIGNALS, MinuteConfig, config_fingerprint
from awsad.station_health import station_health

MINUTE_NS = 60_000_000_000
PATTERNS = {
    "forecast_residual": ("short-horizon forecast mismatch", "Compare the original report with adjacent readings and logger records."),
    "abrupt_change": ("abrupt step or spike", "Check subsequent recovery, sensor connection and logger record; weather can also change abruptly."),
    "flatline_minutes": ("prolonged exact repetition", "Check sensor resolution, data refresh, logger connectivity and whether the environment was stable."),
    "hour_residual": ("long-horizon forecast mismatch", "Compare the longer weather trend and calibration records before attributing a sensor cause."),
    "sustained_deviation": ("persistent signed forecast mismatch", "Review sustained weather change and calibration history; this signal alone cannot establish drift."),
}


def _timestamp(value, *, minute_aligned=True) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be a timezone-aware timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("timestamp must be a timezone-aware timestamp")
    timestamp = timestamp.tz_convert("UTC")
    try:
        nanoseconds = timestamp.value
    except OverflowError as exc:
        raise ValueError("timestamp is outside the supported nanosecond range") from exc
    if minute_aligned and nanoseconds % MINUTE_NS:
        raise ValueError("the frozen native-minute detector requires minute-aligned timestamps")
    return timestamp


def json_safe(value):
    """Finite JSON serialization, preserving missing values as null."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT or value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


@dataclass
class _GroupState:
    history: OrderedDict = field(default_factory=OrderedDict)
    ids: dict = field(default_factory=dict)
    flat: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    health: deque = field(default_factory=deque)
    latest: dict | None = None
    rows_seen: int = 0
    unreported_slots: int = 0
    advanced_to: pd.Timestamp | None = None


class LiveMinuteDetector:
    """Thread-safe per-station|source incremental detector.

    Strict timestamp order rejects old observations. IDs are also deduplicated
    within bounded retained context, not an unbounded lifetime registry. The
    optional pattern predictor receives bounded raw history and a scored record.
    """

    def __init__(self, models, thresholds, config: MinuteConfig, *,
                 expected_cadence_minutes: int | None = None, health_window_rows: int = 1440,
                 max_groups: int = 16, allowed_group: str | None = None,
                 pattern_predictor: Callable | None = None, model_info: dict | None = None):
        config.validate()
        if expected_cadence_minutes not in (None, 1) or isinstance(expected_cadence_minutes, bool):
            raise ValueError("native-minute cadence must be 1 minute or unspecified")
        if not 1 <= health_window_rows <= 10080 or not 1 <= max_groups <= 64:
            raise ValueError("health window must be 1..10080 rows and groups 1..64")
        if "pooled_seen_groups" not in thresholds:
            raise ValueError("frozen pooled thresholds are required")
        for group_thresholds in thresholds.values():
            for channel in CHANNELS:
                for signal in SIGNALS:
                    threshold = group_thresholds[channel][signal]["threshold"]
                    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
                        raise ValueError("frozen thresholds must be finite and nonnegative")
        if set(models) != {f"{h}:{c}" for h in config.horizons for c in CHANNELS}:
            raise ValueError("frozen model keys do not match the minute schema")
        self.models, self.thresholds, self.config = models, thresholds, config
        self.expected_cadence_minutes = expected_cadence_minutes
        self.health_window_rows, self.max_groups = health_window_rows, max_groups
        self.allowed_group, self.pattern_predictor = allowed_group, pattern_predictor
        self.history_limit = max(181, max(config.horizons) + max(config.context_minutes) + 1)
        self.model_info = model_info or {"config_sha256": config_fingerprint(config)}
        self._groups: dict[str, _GroupState] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_artifacts(cls, directory: str | Path, **kwargs):
        """Load trusted local artifacts; model paths must not come from HTTP input."""
        root = Path(directory)
        detector_bytes = (root / "detector.json").read_bytes()
        detector = json.loads(detector_bytes)
        if detector.get("dataset_policy") != "real_observations_only":
            raise ValueError("live scoring requires the real-observation minute detector")
        config_data = dict(detector["config"])
        for key in ("holdout_station_ids", "horizons", "context_minutes"):
            config_data[key] = tuple(config_data[key])
        config = MinuteConfig(**config_data)
        if detector.get("config_sha256") != config_fingerprint(config):
            raise ValueError("frozen config fingerprint mismatch")
        model_path = root / "models.joblib"
        return cls(joblib.load(model_path), detector["thresholds"], config,
                   model_info={"config_sha256": config_fingerprint(config),
                               "detector_sha256": hashlib.sha256(detector_bytes).hexdigest(),
                               "models_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                               "source": "frozen real-observation minute detector",
                               "hardware_fault_accuracy": None}, **kwargs)

    def _validate(self, observation: Mapping) -> tuple[dict, str, pd.Timestamp]:
        if not isinstance(observation, Mapping):
            raise ValueError("each observation must be an object")
        row = dict(observation)
        required = {"timestamp", "observation_id", "station_id", "source", *CHANNELS}
        if required - row.keys():
            raise ValueError("timestamp, original identity, station/source and all three nullable channels are required")
        for key in ("observation_id", "station_id", "source"):
            if not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > 256:
                raise ValueError(f"{key} must be a nonempty string of at most 256 characters")
        if "|" in row["station_id"] or "|" in row["source"]:
            raise ValueError("station/source identifiers cannot contain '|'")
        group = row["station_id"] + "|" + row["source"]
        if self.allowed_group is not None and group != self.allowed_group:
            raise ValueError("observation group does not match this session")
        if "group" in row and row["group"] != group:
            raise ValueError("group does not match station_id|source")
        label = row.get("label")
        if label is not None and label is not pd.NA and not (isinstance(label, float) and math.isnan(label)):
            raise ValueError("hardware fault labels must remain unknown in the observation stream")
        for channel in CHANNELS:
            value = row[channel]
            if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
                row[channel] = None
            elif isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.number)) or not math.isfinite(value):
                raise ValueError(f"{channel} must be a finite number or null")
            else:
                row[channel] = float(value)
            row.setdefault(channel + "__qc_accepted", None)
        timestamp = _timestamp(row["timestamp"])
        row["timestamp"] = timestamp
        return row, group, timestamp

    def ingest(self, observation: Mapping) -> dict:
        with self._lock, threadpool_limits(limits=self.config.max_cpu_threads):
            row, group, timestamp = self._validate(observation)
            return self._ingest_validated(row, group, timestamp)

    def ingest_many(self, observations) -> list[dict]:
        """Validate a complete packet before consuming rows, preserving input order."""
        if not isinstance(observations, (list, tuple)) or not 1 <= len(observations) <= 360:
            raise ValueError("a packet must contain 1..360 observations")
        with self._lock, threadpool_limits(limits=self.config.max_cpu_threads):
            validated = [self._validate(row) for row in observations]
            last = {g: s.latest["timestamp"] for g, s in self._groups.items()}
            identities = {g: set(s.ids) for g, s in self._groups.items()}
            for row, group, timestamp in validated:
                if group in last and timestamp <= last[group]:
                    raise ValueError("duplicate or out-of-order observation timestamp")
                if row["observation_id"] in identities.setdefault(group, set()):
                    raise ValueError("duplicate observation ID in retained context or packet")
                last[group] = timestamp
                identities[group].add(row["observation_id"])
            if len(last) > self.max_groups:
                raise ValueError("session station capacity reached")
            return [self._ingest_validated(*item) for item in validated]

    def _ingest_validated(self, row, group, timestamp):
        state = self._groups.get(group)
        if state is None:
            if len(self._groups) >= self.max_groups:
                raise ValueError("session station capacity reached")
            state = _GroupState(health=deque(maxlen=self.health_window_rows),
                                residuals={c: deque(maxlen=self.config.sustained_minutes) for c in CHANNELS})
            self._groups[group] = state
        previous = state.latest
        if previous is not None and timestamp <= previous["timestamp"]:
            raise ValueError("duplicate or out-of-order observation timestamp")
        if row["observation_id"] in state.ids:
            raise ValueError("duplicate observation ID in retained context")
        contiguous = previous is not None and timestamp - previous["timestamp"] == pd.Timedelta(minutes=1)
        if not contiguous:
            for values in state.residuals.values():
                values.clear()
        predictions = {}
        for horizon in self.config.horizons:
            blocks = []
            for lag in self.config.context_minutes:
                past = state.history.get(timestamp.value - (horizon + lag) * MINUTE_NS)
                blocks.append(np.array([past[c] if past is not None and past[c] is not None else np.nan
                                        for c in CHANNELS], dtype=float))
            features = np.concatenate(blocks + [blocks[0] - b for b in blocks[1:]])
            predicted = np.full(3, np.nan)
            if np.isfinite(features).all():
                for i, channel in enumerate(CHANNELS):
                    model = self.models[f"{horizon}:{channel}"]
                    predicted[i] = blocks[0][i] + (float(model.predict(features[None, :])[0]) if model is not None else 0.)
            predictions[horizon] = predicted
        threshold_group = group if group in self.thresholds else "pooled_seen_groups"
        out = dict(row)
        out.update(group=group, threshold_group=threshold_group, hardware_fault_status="unknown")
        reasons, explanations, physics_alerts = [], [], []
        all_scores, any_candidate = [], False
        for i, channel in enumerate(CHANNELS):
            value = row[channel] if row[channel] is not None else np.nan
            prior = previous[channel] if previous is not None and previous[channel] is not None else np.nan
            same = contiguous and math.isfinite(value) and value == prior
            flat = state.flat.get(channel, 0.) + 1 if same else (0. if math.isfinite(value) else np.nan)
            state.flat[channel] = flat
            signed = value - predictions[60][i]
            residuals = state.residuals[channel]
            residuals.append(signed)
            sustained = (abs(float(np.mean(residuals))) if len(residuals) == self.config.sustained_minutes
                         and np.isfinite(residuals).all() else np.nan)
            signals = {"forecast_residual": abs(value - predictions[1][i]),
                       "abrupt_change": abs(value - prior) if contiguous else np.nan,
                       "flatline_minutes": flat, "hour_residual": abs(signed),
                       "sustained_deviation": sustained}
            channel_scores, channel_reasons = [], []
            for signal, signal_value in signals.items():
                out[f"{channel}__{signal}"] = signal_value
                if not math.isfinite(signal_value):
                    continue
                threshold = self.thresholds[threshold_group][channel][signal]["threshold"]
                ratio = signal_value / threshold if threshold > 0 else (2. if signal_value > 0 else 0.)
                channel_scores.append(ratio)
                if signal_value > threshold:
                    channel_reasons.append(signal)
                    reasons.append(channel + ":" + signal)
                    pattern, action = PATTERNS[signal]
                    explanations.append({"channel": channel, "signal": signal, "pattern": pattern,
                                         "reason": f"{signal_value:.6g} exceeds frozen reference threshold {threshold:.6g}",
                                         "value": signal_value, "threshold": threshold, "exceedance_ratio": ratio,
                                         "action": action, "confidence": None,
                                         "confidence_semantics": "uncalibrated pattern suggestion; no hardware probability"})
            score = max(channel_scores) if channel_scores else None
            out.update({f"{channel}__prediction": predictions[1][i],
                        f"{channel}__prediction_60m": predictions[60][i],
                        f"{channel}__score": score, f"{channel}__reason_codes": "|".join(channel_reasons),
                        f"{channel}__alert": bool(channel_reasons) if channel_scores else None})
            if score is not None:
                all_scores.append(score)
            any_candidate |= bool(channel_reasons)
            invalid = (math.isfinite(value) and (
                (channel == "relative_humidity_pct" and not 0 <= value <= 100)
                or (channel == "pressure_hpa" and value <= 0)
                or (channel == "temperature_c" and value < -273.15)))
            if invalid:
                signal, reason = "physical_domain", "Temperature is below absolute zero in the stated Celsius units."
                if channel == "relative_humidity_pct":
                    signal = "reporting_range" if value > 100 else "physical_domain"
                    reason = ("RH exceeds the conventional 0–100% reporting range; supersaturation or measurement/reporting effects require source-specific review."
                              if value > 100 else "Negative relative humidity is outside its physical definition.")
                elif channel == "pressure_hpa":
                    signal, reason = "reporting_range", "Nonpositive station pressure is outside the expected atmospheric-station reporting domain."
                physics_alerts.append({"channel": channel, "signal": signal,
                                       "reason": reason,
                                       "action": "Check units, parsing, source report and sensor/logger configuration.",
                                       "hardware_fault_status": "unknown"})
        gap_slots = (max(0, int((timestamp - previous["timestamp"]) // pd.Timedelta(minutes=1)) - 1)
                     if self.expected_cadence_minutes is not None and previous is not None else
                     0 if self.expected_cadence_minutes is not None else None)
        if gap_slots:
            state.unreported_slots += gap_slots
        availability = {"missing_channels": [c for c in CHANNELS if row[c] is None],
                        "unreported_slots_before": gap_slots,
                        "expected_cadence_minutes": self.expected_cadence_minutes,
                        "semantics": "Data availability only; cause unknown. No values were filled."}
        score = max(all_scores) if all_scores else None
        out.update(anomaly_score=score, is_candidate=any_candidate, reason_codes="|".join(reasons),
                   scoring_status="scored_available_signals" if all_scores else "missing_observation",
                   severity="high" if physics_alerts or (score is not None and score >= 2 and any_candidate) else
                            "review" if any_candidate else "info",
                   explanations=explanations, physics_alerts=physics_alerts, availability=availability,
                   type_confidence=None, fault_type=None)
        # Preserve source metadata in the returned record, but never accumulate
        # arbitrary payloads in model context. The hook sees observed inputs only.
        context_keys = ["timestamp", "observation_id", "station_id", "source", *CHANNELS]
        state.history[timestamp.value] = {key: row[key] for key in context_keys}
        state.ids[row["observation_id"]] = timestamp.value
        while len(state.history) > self.history_limit or (
                state.history and next(iter(state.history)) < timestamp.value - (self.history_limit - 1) * MINUTE_NS):
            _, removed = state.history.popitem(last=False)
            state.ids.pop(removed["observation_id"], None)
        if self.pattern_predictor is not None:
            try:
                out["pattern_evidence"] = self.pattern_predictor(pd.DataFrame(state.history.values()), dict(out))
            except Exception as exc:
                out["pattern_evidence"] = {"status": "unavailable", "reason": type(exc).__name__}
        state.latest = out
        state.rows_seen += 1
        health_keys = ["timestamp", *CHANNELS, *(c + "__alert" for c in CHANNELS),
                       *(c + "__reason_codes" for c in CHANNELS)]
        state.health.append({key: out[key] for key in health_keys})
        return json_safe(out)

    def advance(self, timestamp, group: str | None = None) -> list[dict]:
        """Check expected minute slots without inventing rows (no internal timer).

        Supply current time for a connected feed, or replay time for replay. A
        later original observation may resolve a previously overdue slot.
        """
        now = _timestamp(timestamp, minute_aligned=False)
        with self._lock:
            groups = {group: self._groups[group]} if group in self._groups else self._groups if group is None else {}
            if group is not None and not groups:
                raise ValueError("unknown station group")
            if any(now < state.latest["timestamp"] or
                   (state.advanced_to is not None and now < state.advanced_to)
                   for state in groups.values()):
                raise ValueError("heartbeat cannot move backwards")
            notices = []
            for key, state in groups.items():
                state.advanced_to = now
                overdue = (int((now - state.latest["timestamp"]) // pd.Timedelta(minutes=1))
                           if self.expected_cadence_minutes is not None else None)
                notices.append({"group": key, "kind": "data_availability", "checked_at": now.isoformat(),
                                "last_observation_at": state.latest["timestamp"].isoformat(),
                                "expected_cadence_minutes": self.expected_cadence_minutes,
                                "overdue_slots": overdue,
                                "state": "overdue" if overdue else "unknown_cadence" if overdue is None else "current",
                                "hardware_fault_status": "unknown",
                                "action": "Check feed transport, power and logger status; no hardware cause is established." if overdue else None})
            return notices

    def snapshot(self) -> dict:
        with self._lock:
            groups = {}
            for group, state in self._groups.items():
                groups[group] = {"rows_seen": state.rows_seen, "last_timestamp": state.latest["timestamp"],
                                 "buffered_observations": len(state.history),
                                 "health": station_health(pd.DataFrame(state.health), expected_cadence_minutes=self.expected_cadence_minutes),
                                 "latest": state.latest,
                                 "availability": {"unreported_slots_since_start": state.unreported_slots
                                                  if self.expected_cadence_minutes is not None else None}}
            return json_safe({"groups": groups, "model_info": self.model_info,
                              "limits": {"max_groups": self.max_groups, "history_rows_per_group": self.history_limit,
                                         "health_rows_per_group": self.health_window_rows, "max_packet_rows": 360},
                              "semantics": "Incremental inference; historical inputs remain historical replay. Candidates are review proposals."})
