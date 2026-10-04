"""Separate, labelled software scenarios for the incremental dashboard feed.

This demonstration never edits an original row, trains a model, or assigns a
hardware-fault label. It keeps archival timestamps and missing values intact.
Scenario markers identify applied numeric changes, not detector decisions.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from awsad.live_detector import json_safe
from awsad.minute_detection import CHANNELS

VERSION = "operational-synthetic-feed-v1"
SOURCE = "synthetic_demo"
WARMUP_MINUTES = 180
SCENARIOS = ("spike", "drift", "stuck", "dropout", "bias", "noise_burst", "clipping", "scale_error", "sensor_swap")
DEFAULT_DURATIONS = {"spike": 3, "bias": 60, "drift": 120, "stuck": 150,
                     "dropout": 30, "noise_burst": 30, "clipping": 90,
                     "scale_error": 90, "sensor_swap": 45}
NOTICE = "SYNTHETIC software scenario over unchanged historical observations; not a connected station or confirmed hardware fault."


def prepare_operational_feed(frame: pd.DataFrame, *, scenario="spike", channel="temperature_c", seed=26073) -> dict:
    """Return synthetic packets plus full original baselines for a <=6h window.

    The first 180 minutes are unmodified. A scenario starts only after 180 exact
    preceding minute records; original missing inputs never gain invented values.
    This independent demonstration generator is not the benchmark training path.
    """
    if scenario not in SCENARIOS or channel not in CHANNELS:
        raise ValueError("choose a supported scenario and one of the three measured channels")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer from 0 through 4294967295")
    required = {"timestamp", "observation_id", "station_id", "source", *CHANNELS}
    if not required <= set(frame) or not WARMUP_MINUTES < len(frame) <= 360:
        raise ValueError("Use a six-hour baseline window: 180 warmup minutes must be followed by eligible original readings.")
    if frame[["station_id", "source"]].drop_duplicates().shape[0] != 1:
        raise ValueError("one original station|source group is required")
    station, source = str(frame.station_id.iloc[0]), str(frame.source.iloc[0])
    if not station or not source or "|" in station or "|" in source or source == SOURCE:
        raise ValueError("the baseline must be one original station source, not a synthetic feed")
    if frame.observation_id.isna().any() or frame.observation_id.duplicated().any():
        raise ValueError("baseline observation identities must be present and unique")
    if "label" in frame and frame.label.notna().any():
        raise ValueError("hardware labels must remain unknown in this demonstration baseline")
    if "hardware_fault_status" in frame and frame.hardware_fault_status.dropna().ne("unknown").any():
        raise ValueError("baseline hardware status must remain unknown for this demonstration")
    if "synthetic" in frame and frame.synthetic.fillna(False).any():
        raise ValueError("synthetic observations cannot be reused as the original baseline")
    times = pd.to_datetime(frame.timestamp, errors="raise")
    if not isinstance(times.dtype, pd.DatetimeTZDtype) or times.isna().any():
        raise ValueError("baseline timestamps must be timezone aware")
    times = times.dt.tz_convert("UTC")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("baseline timestamps must be unique and chronological")
    ticks = pd.DatetimeIndex(times).as_unit("ns").asi8
    if np.any(ticks % 60_000_000_000) or times.iloc[-1] - times.iloc[0] >= pd.Timedelta(hours=6):
        raise ValueError("baseline must contain native minute-aligned rows in at most six hours")
    original = frame[list(CHANNELS)].to_numpy(dtype=float, na_value=np.nan, copy=True)
    if np.isinf(original).any():
        raise ValueError("original measurements must be finite or missing")
    baseline_records = json_safe(frame.to_dict("records"))
    baseline_group = station + "|" + source
    if "group" in frame and frame.group.ne(baseline_group).any():
        raise ValueError("baseline group differs from its station and source")

    target = CHANNELS.index(channel)
    affected = [0, 2] if scenario == "sensor_swap" else [target]
    start_index = None
    for index in range(WARMUP_MINUTES, len(frame)):
        if not np.all(np.diff(ticks[index - WARMUP_MINUTES:index + 1]) == 60_000_000_000):
            continue
        if not np.isfinite(original[index, affected]).all():
            continue
        if scenario == "stuck" and not np.isfinite(original[index - 1, target]):
            continue
        start_index = index
        break
    if start_index is None:
        raise ValueError("Use a six-hour baseline with 180 consecutive warmup minutes and an available target channel; no eligible intervention was found.")
    start = times.iloc[start_index]
    available_minutes = int((times.iloc[-1] + pd.Timedelta(minutes=1) - start) / pd.Timedelta(minutes=1))
    recovery_minutes = min(30, available_minutes // 3)
    duration = min(DEFAULT_DURATIONS[scenario], max(1, available_minutes - recovery_minutes))
    if scenario == "drift" and duration < 2:
        raise ValueError("Use a six-hour baseline; a drift needs at least two eligible minutes after warmup.")
    end = start + pd.Timedelta(minutes=duration)
    active = times.ge(start).to_numpy() & times.lt(end).to_numpy()
    positions = np.flatnonzero(active)
    baseline_hash = hashlib.sha256(json.dumps(baseline_records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    identity = f"{VERSION}|{baseline_hash}|{scenario}|{channel}|{seed}"
    event_hash = hashlib.sha256(identity.encode()).hexdigest()
    rng = np.random.default_rng(int(event_hash[:16], 16))
    sign = float(rng.choice((-1., 1.)))
    magnitude = float((8., 12., 25.)[target] * sign)
    after = original.copy()
    values = original[active, target].copy()
    finite = np.isfinite(values)
    parameters = {"duration_minutes": duration, "recovery_minutes_reserved": recovery_minutes}
    if scenario in ("spike", "bias"):
        values[finite] += magnitude
        parameters["offset"] = magnitude
    elif scenario == "drift":
        elapsed = ((times.iloc[positions] - start) / pd.Timedelta(minutes=1)).to_numpy(float)
        ramp = (elapsed + 1.) / duration
        values[finite] += magnitude * ramp[finite]
        parameters.update(final_offset=magnitude, ramp="linear in elapsed minutes")
    elif scenario == "stuck":
        latch = float(original[start_index - 1, target])
        values[finite] = latch
        parameters["latched_value"] = latch
    elif scenario == "dropout":
        values[finite] = np.nan
    elif scenario == "noise_burst":
        sigma = abs(magnitude) * .45
        values[finite] += rng.normal(0., sigma, len(values))[finite]
        parameters["noise_standard_deviation"] = sigma
    elif scenario == "clipping":
        # A declared limit based on the first affected original value, not on
        # model outcomes or future observations. It guarantees a visible test.
        upper = sign > 0
        bound = float(original[start_index, target] - abs(magnitude) / 2 if upper
                      else original[start_index, target] + abs(magnitude) / 2)
        values[finite] = np.minimum(values[finite], bound) if upper else np.maximum(values[finite], bound)
        parameters.update(bound=bound, direction="upper" if upper else "lower")
    elif scenario == "scale_error":
        factor = (1.35, 1.02, 1.3)[target]
        if sign < 0:
            factor = 1. / factor
        values[finite] *= factor
        parameters["gain"] = factor
    elif scenario == "sensor_swap":
        joint = active & np.isfinite(original[:, 0]) & np.isfinite(original[:, 2])
        after[joint, 0], after[joint, 2] = original[joint, 2], original[joint, 0]
        parameters["operation"] = "numeric temperature/RH channel exchange without unit conversion"
    if scenario != "sensor_swap":
        after[active, target] = values
    if np.any(np.isnan(original) & np.isfinite(after)):
        raise AssertionError("demonstration attempted to fill a missing original")
    modified = ~((after == original) | (np.isnan(after) & np.isnan(original)))
    changed_rows = np.flatnonzero(modified.any(axis=1))
    if not len(changed_rows):
        raise ValueError("This baseline produced no numeric modification; choose a different six-hour window or channel.")
    group = station + "|" + SOURCE
    scenario_id = "synthetic-demo:" + event_hash[:24]
    event = {"scenario_id": scenario_id, "version": VERSION, "scenario": scenario,
             "channels": [CHANNELS[index] for index in affected], "requested_channel": channel,
             "scheduled_start": start.isoformat(), "scheduled_end_exclusive": end.isoformat(),
             "first_modified": times.iloc[changed_rows[0]].isoformat(),
             "last_modified": times.iloc[changed_rows[-1]].isoformat(),
             "modified_rows": int(len(changed_rows)), "modified_values": int(modified.sum()),
             "parameters": parameters, "baseline_window_sha256": baseline_hash,
             "baseline_hash_semantics": "SHA256 of canonical selected original-row JSON; raw-file byte hashes remain in baseline records",
             "semantics": "Applied software modification, not model detection or confirmed hardware cause."}
    packets = []
    for index, baseline in enumerate(baseline_records):
        modified_channels = [c for i, c in enumerate(CHANNELS) if modified[index, i]]
        packet = {"timestamp": times.iloc[index].isoformat(), "station_id": station, "source": SOURCE,
                  "group": group, "observation_id": f"synthetic:{event_hash[:24]}:{index:04d}",
                  "synthetic": True, "mode": SOURCE, "label": None, "hardware_fault_status": "unknown",
                  "eligible_for_real_training": False, "benchmark_evaluation": False,
                  "scenario_id": scenario_id, "scenario_label": scenario if modified_channels else "no_injection",
                  "scenario_modified": bool(modified_channels), "scenario_modified_channels": modified_channels,
                  "scenario_notice": NOTICE, "scenario_metadata": {"scenario": scenario, "version": VERSION,
                      "marker_semantics": "software intervention only; not a detector input or hardware label"},
                  "baseline": baseline, "baseline_observation_id": baseline["observation_id"],
                  "baseline_source": source, "baseline_group": baseline_group,
                  "baseline_raw_file_sha256": baseline.get("raw_file_sha256"),
                  "baseline_raw_file_name": baseline.get("raw_file_name"),
                  "baseline_raw_row_number": baseline.get("raw_row_number"),
                  "baseline_source_url": baseline.get("source_url"),
                  "baseline_provider": {c: baseline.get(c + "__provider") for c in CHANNELS},
                  "threshold_reference": "pooled_seen_groups"}
        for i, name in enumerate(CHANNELS):
            packet[name] = float(after[index, i]) if np.isfinite(after[index, i]) else None
            packet["baseline_" + name] = baseline[name]
            packet[name + "__scenario_modified"] = bool(modified[index, i])
            # Provider acceptance belongs to the original source row, not its
            # synthetic copy. Original raw QC remains in the nested baseline.
            packet[name + "__qc_accepted"] = None
        packets.append(packet)
    return {"mode": SOURCE, "group": group, "baseline_group": baseline_group,
            "scenario": scenario, "channel": channel, "seed": int(seed), "rows": len(packets),
            "start": times.iloc[0].isoformat(), "end": (times.iloc[-1] + pd.Timedelta(minutes=1)).isoformat(),
            "observations": packets, "event": event, "threshold_reference": "pooled_seen_groups",
            "threshold_semantics": "Synthetic source uses the frozen pooled reference; no real-station source is impersonated.",
            "warmup_minutes": {"scenario_context": WARMUP_MINUTES, "forecast_1m": 61,
                               "forecast_60m": 120, "sustained": 149},
            "semantics": NOTICE, "eligible_for_real_training": False,
            "benchmark_evaluation": False, "hardware_fault_status": "unknown"}
