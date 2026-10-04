"""Preliminary three-channel edge watchdog, independent of the hub ML model.

This module uses only portable Python syntax and the standard math module.
No observations, predictions, or hardware-fault labels are fabricated. Keep the
original packet and provenance in the caller's log; this fixed-state helper
does not retain an observation archive or implement a sensor/radio driver.
"""

import math


CHANNELS = ("temperature_c", "pressure_hpa", "relative_humidity_pct")
VERSION = "preliminary-edge-watchdog-v2"
MAX_COUNTER = 2147483647


def _number(value, name, allow_missing=False):
    if value is None and allow_missing:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a finite number" + (" or None" if allow_missing else ""))
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(name + " is outside the supported numeric range")
    if not math.isfinite(result):
        raise ValueError(name + " must be finite; represent a missing reading as None")
    return result


class PreliminaryWatchdog:
    """One station|source, three channels, constant-size state.

    Step limits are explicit operator policies in C, hPa, and percentage points
    per adjacent sample; the defaults are not trained or validated thresholds.
    Flatline checks compare against the run's first value, preventing gradual
    epsilon-sized drift from being counted as an unchanged run.
    """

    __slots__ = ("station_source", "cadence_seconds", "grace_seconds", "step_limits",
                 "flatline_seconds", "flatline_epsilon", "policy_origin", "last_timestamp",
                 "last_heartbeat", "rows_seen", "last_values", "flatline_anchors", "flatline_starts")

    def __init__(self, station_source, cadence_seconds, step_limits=None,
                 flatline_seconds=None, flatline_epsilon=0.0, grace_seconds=0.0, policy_origin=None):
        if not isinstance(station_source, str) or not 3 <= len(station_source) <= 128:
            raise ValueError("station_source must be a station|source string of at most 128 characters")
        if station_source.count("|") != 1 or not all(station_source.split("|")):
            raise ValueError("station_source must identify exactly one station|source")
        cadence = _number(cadence_seconds, "cadence_seconds")
        grace = _number(grace_seconds, "grace_seconds")
        defaults = step_limits is None and flatline_seconds is None and flatline_epsilon == 0
        step_limits = (5.0, 6.0, 20.0) if step_limits is None else step_limits
        flatline_seconds = 600.0 if flatline_seconds is None else flatline_seconds
        flat = flatline_seconds if isinstance(flatline_seconds, (tuple, list)) else (flatline_seconds,) * 3
        if len(flat) != 3:
            raise ValueError("flatline_seconds must be a scalar or three per-channel thresholds")
        flat = tuple(_number(value, "flatline_seconds") for value in flat)
        epsilon = _number(flatline_epsilon, "flatline_epsilon")
        if cadence <= 0 or not 0 <= grace < cadence or any(value < 0 for value in flat) or epsilon < 0:
            raise ValueError("require positive cadence, 0 <= grace < cadence, nonnegative flatline thresholds and epsilon")
        if len(step_limits) != 3:
            raise ValueError("exactly three step limits are required")
        limits = tuple(_number(value, "step_limit") for value in step_limits)
        if any(value < 0 for value in limits):
            raise ValueError("step limits must be nonnegative")
        origin = policy_origin or ("default_operator_policy" if defaults else "provided_operator_policy")
        if origin not in ("default_operator_policy", "provided_operator_policy", "frozen_signal_threshold_export"):
            raise ValueError("unknown policy_origin")
        self.station_source = station_source
        self.cadence_seconds, self.grace_seconds = cadence, grace
        self.step_limits, self.flatline_seconds, self.flatline_epsilon = limits, flat, epsilon
        self.policy_origin = origin
        self.last_timestamp = self.last_heartbeat = None
        self.rows_seen = 0
        self.last_values = [None, None, None]
        self.flatline_anchors = [None, None, None]
        self.flatline_starts = [None, None, None]

    def ingest(self, timestamp_s, values, station_source=None):
        """Score an actual packet; time is nonnegative seconds in one clock epoch.

        Reject nonfinite values, mixed station streams and non-increasing packet
        times before changing state. Absent channel keys and None remain missing.
        A heartbeat does not prohibit a delayed but ordered original packet.
        """
        timestamp = _number(timestamp_s, "timestamp_s")
        if not 0 <= timestamp <= 1e12:
            raise ValueError("timestamp_s must be between 0 and 1e12 seconds")
        if station_source is not None and station_source != self.station_source:
            raise ValueError("packet station|source differs from this watchdog")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("packet timestamps must be strictly increasing")
        if not isinstance(values, dict):
            raise ValueError("values must be a mapping of the three observed channels")
        readings = [_number(values.get(channel), channel, True) for channel in CHANNELS]
        elapsed = None if self.last_timestamp is None else timestamp - self.last_timestamp
        adjacent = elapsed is not None and abs(elapsed - self.cadence_seconds) <= self.grace_seconds
        skipped = 0 if elapsed is None else max(0, int(elapsed / self.cadence_seconds) - 1)
        last_values, anchors, starts, channels, alerts = [], [], [], {}, []
        for index, channel in enumerate(CHANNELS):
            value = readings[index]
            present = value is not None
            invalid = present and ((index == 0 and value < -273.15) or (index == 2 and value < 0))
            reporting_range = present and ((index == 1 and value <= 0) or (index == 2 and value > 100))
            usable = present and not invalid and not reporting_range
            temporal = usable and adjacent and self.last_values[index] is not None
            delta = abs(value - self.last_values[index]) if temporal else None
            step = delta > self.step_limits[index] if temporal else None
            if usable:
                same_run = temporal and abs(value - self.flatline_anchors[index]) <= self.flatline_epsilon
                anchor = self.flatline_anchors[index] if same_run else value
                run_start = self.flatline_starts[index] if same_run else timestamp
                duration = timestamp - run_start
                repeated = duration > self.flatline_seconds[index] if temporal else None
            else:
                anchor, run_start, duration, repeated = None, None, None, None
            channels[channel] = {
                "observed": value, "availability": "observed" if present else "missing",
                "hardware_fault_status": "unknown", "physical_domain_alert": invalid if present else None,
                "reporting_range_alert": reporting_range if present else None,
                "step_alert": step, "flatline_alert": repeated,
                "step_magnitude": delta if delta is None or math.isfinite(delta) else None,
                "flatline_duration_seconds": duration,
                "signal_availability": {"physical_domain": present, "reporting_range": present, "step": temporal, "flatline": temporal},
            }
            if invalid:
                alerts.append({"channel": channel, "signal": "physical_domain", "priority": "review_now",
                               "reason": "Temperature is below absolute zero in Celsius." if index == 0 else "Negative RH is outside its physical definition.",
                               "action": "Check units, decoding, original report and sensor/logger configuration."})
            if reporting_range:
                alerts.append({"channel": channel, "signal": "reporting_range", "priority": "review_candidates",
                               "reason": "RH exceeds the conventional 0-100% reporting range; supersaturation or measurement/reporting effects require source-specific review."
                                         if index == 2 else "Nonpositive station pressure is outside the expected atmospheric-station reporting domain.",
                               "action": "Check units, original source report and sensor/logger configuration; no hardware cause is established."})
            if step:
                alerts.append({"channel": channel, "signal": "step", "priority": "review_candidates",
                               "action": "Inspect adjacent reports and recovery; weather can also change abruptly."})
            if repeated:
                alerts.append({"channel": channel, "signal": "flatline", "priority": "review_candidates",
                               "action": "Check sensor resolution, data refresh and whether the environment was stable."})
            last_values.append(value if usable else None)
            anchors.append(anchor)
            starts.append(run_start)
        self.last_timestamp = timestamp
        self.rows_seen = min(MAX_COUNTER, self.rows_seen + 1)
        self.last_values, self.flatline_anchors, self.flatline_starts = last_values, anchors, starts
        return {"version": VERSION, "group": self.station_source, "timestamp_s": timestamp, "policy_origin": self.policy_origin,
                "rows_seen": self.rows_seen, "channels": channels, "alerts": alerts,
                "is_candidate": bool(alerts), "hardware_fault_status": "unknown",
                "availability": {"missing_channels": [CHANNELS[i] for i in range(3) if readings[i] is None],
                                 "unreported_slots_before": skipped, "cause": "unknown"},
                "semantics": "Preliminary policy watchdog; not hub ML, hardware diagnosis, or failure-time prediction."}

    def heartbeat(self, timestamp_s):
        """Check packet silence using an independently supplied receiver clock.

        Repeated heartbeats report the current overdue count, not additive missing
        observations. No gaps can be inferred before the first actual packet.
        """
        now = _number(timestamp_s, "timestamp_s")
        if not 0 <= now <= 1e12:
            raise ValueError("timestamp_s must be between 0 and 1e12 seconds")
        if ((self.last_timestamp is not None and now < self.last_timestamp)
                or (self.last_heartbeat is not None and now < self.last_heartbeat)):
            raise ValueError("heartbeat clock cannot go backwards")
        self.last_heartbeat = now
        if self.last_timestamp is None:
            state, overdue = "waiting_for_first_packet", None
        else:
            overdue = max(0, int((now - self.last_timestamp - self.grace_seconds) / self.cadence_seconds))
            state = "packet_overdue" if overdue else "no_packet_overdue"
        return {"version": VERSION, "group": self.station_source, "checked_at_s": now,
                "state": state, "unreported_slots": overdue, "hardware_fault_status": "unknown",
                "action": "Check transport, power and logger status; no hardware cause is established." if overdue else None,
                "semantics": "Packet availability only; this check neither scores nor invents observations."}

    def snapshot(self):
        """A bounded copy of the current policy and state; no historical archive."""
        return {"version": VERSION, "group": self.station_source, "cadence_seconds": self.cadence_seconds,
                "grace_seconds": self.grace_seconds, "step_limits": self.step_limits,
                "flatline_seconds": self.flatline_seconds, "flatline_epsilon": self.flatline_epsilon,
                "policy_origin": self.policy_origin, "comparison": "strictly_greater",
                "last_timestamp": self.last_timestamp, "last_heartbeat": self.last_heartbeat,
                "rows_seen": self.rows_seen, "last_values": self.last_values[:],
                "flatline_anchors": self.flatline_anchors[:], "flatline_starts": self.flatline_starts[:],
                "state_bounds": {"station_groups": 1, "group_max_characters": 128, "channels": 3,
                                 "retained_channel_scalars": 9, "historical_packet_buffer": 0,
                                 "counter_max": MAX_COUNTER}}
