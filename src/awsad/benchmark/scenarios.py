"""Versioned software interventions on separate copies of original observations.

Scenario labels identify modifications, not real hardware truth. All variants of
one source window belong to one partition. Missing originals are never filled.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from awsad.data.acquisition import sha256_file

CHANNELS = ("temperature_c", "pressure_hpa", "relative_humidity_pct")
FAULTS = ("spike", "bias", "drift", "stuck", "dropout", "noise_burst",
          "clipping", "scale_error", "sensor_swap")
VERSION = "sih-original-window-scenarios-v2"
HISTORY = 180
WINDOW = 720


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def partition(station: str, timestamp) -> str | None:
    time = pd.Timestamp(timestamp)
    if station == "gwn":
        return "test_station" if "2025-02" == time.strftime("%Y-%m") else None
    if station not in {"bon", "fpk"}:
        return None
    if time.year == 2023:
        return "fit"
    if time.year == 2024 and time.month <= 8:
        return "selection"
    if time.year == 2024 and time.month in (9, 10):
        return "calibration"
    if time.strftime("%Y-%m") == "2025-02":
        return "test_temporal"
    return None


@dataclass
class SourceWindow:
    frame: pd.DataFrame
    window_id: str
    split: str
    source_file: str
    source_sha256: str


def source_windows(cache_dir, splits, *, windows_per_shard=4, seed=26073):
    """Select disjoint 12-hour clock blocks, never based on detector outcomes.

    Hashes are checked against the verified native cache inventory. Incomplete
    clock blocks are reported as excluded, never interpolated. Missing fields
    within a full block remain missing. A manifest from the original admission
    pipeline is required, not an arbitrary directory of Parquet files.
    """
    root = Path(cache_dir)
    manifest = json.loads((root / "native_manifest.json").read_text())
    if manifest.get("dataset_policy") != "real_observations_only":
        raise ValueError("an original-observation native cache is required")
    if not manifest.get("evidence", {}).get("index_verified"):
        raise ValueError("native source inventory has not been verified")
    for shard in sorted(manifest["shards"], key=lambda item: item["file"]):
        split = partition(shard["station_id"], shard["start"])
        if split not in splits:
            continue
        path = root / shard["file"]
        if path.resolve().parent != root.resolve() or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"source shard hash/path mismatch: {shard['file']}")
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
        if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
            raise ValueError("original timestamps must be unique and chronological")
        known_label = "label" in frame and frame["label"].notna().any()
        known_status = "hardware_fault_status" in frame and frame.hardware_fault_status.ne("unknown").any()
        if frame.observation_id.duplicated().any() or known_label or known_status:
            raise ValueError("original observation identity/unknown-label contract violated")
        blocks = list(frame.groupby(frame.timestamp.dt.floor("12h"), sort=True))
        rng = np.random.default_rng(stable_seed(seed, shard["file"], "windows"))
        order = rng.permutation(len(blocks))[:windows_per_shard]
        for index in sorted(order):
            time, block = blocks[index]
            reason = None
            if len(block) != WINDOW or not block.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
                reason = "incomplete_native_clock_block"
            if reason:
                yield {"excluded": reason, "source_file": shard["file"], "start": time.isoformat(), "split": split}
                continue
            block = block.reset_index(drop=True).copy()
            window_id = f"{shard['station_id']}|{time.isoformat()}"
            yield SourceWindow(block, window_id, split, shard["file"], shard["sha256"])


@dataclass
class Scenario:
    source: SourceWindow
    values: pd.DataFrame
    labels: np.ndarray
    modified: np.ndarray
    event: dict

    def export(self) -> pd.DataFrame:
        """Original full records plus visibly separate simulation columns."""
        out = self.source.frame.copy(deep=True)
        for channel in CHANNELS:
            out["scenario__" + channel] = self.values[channel].to_numpy()
        out["scenario_label"] = self.labels
        out["scenario_modified"] = self.modified.any(axis=1)
        out["scenario_id"] = self.event["scenario_id"]
        out["scenario_partition"] = self.source.split
        out["scenario_notice"] = "SYNTHETIC software intervention; hardware status unknown"
        return out


def generate(source: SourceWindow, fault: str, *, seed=26073, variant=0) -> Scenario:
    if fault not in (*FAULTS, "no_injection"):
        raise ValueError("unknown scenario")
    baseline = source.frame
    values = baseline[["timestamp", *CHANNELS]].copy(deep=True)
    original = values[list(CHANNELS)].to_numpy(float, copy=True)
    after = original.copy()
    rng = np.random.default_rng(stable_seed(seed, source.window_id, fault, variant))
    channel = int(rng.integers(3))
    start = int(rng.integers(HISTORY + 15, HISTORY + 90))
    lengths = {"spike": (1, 5), "bias": (30, 181), "drift": (90, 271),
               "stuck": (30, 241), "dropout": (10, 91), "noise_burst": (10, 91),
               "clipping": (45, 241), "scale_error": (45, 241), "sensor_swap": (30, 181)}
    duration = int(rng.integers(*lengths[fault])) if fault != "no_injection" else 0
    end = min(len(values) - 120, start + duration)
    a, b = np.array([1.5, 2., 6.]), np.array([10., 15., 35.])
    magnitude = float(rng.uniform(a[channel], b[channel]) * rng.choice([-1, 1]))
    affected = [channel]
    params = {"duration_minutes": end - start, "magnitude": magnitude}
    if fault != "no_injection":
        segment = original[start:end, channel].copy()
        finite = np.isfinite(segment)
        if fault in {"spike", "bias"}:
            segment[finite] += magnitude
        elif fault == "drift":
            power = float(rng.uniform(.7, 2.))
            ramp = np.linspace(0, 1, end - start) ** power
            segment[finite] += magnitude * ramp[finite]
            params["ramp_power"] = power
        elif fault == "stuck":
            previous = original[start - 1, channel]
            if np.isfinite(previous):
                segment[finite] = previous
            params["latched_value"] = float(previous) if np.isfinite(previous) else None
        elif fault == "dropout":
            if rng.random() < .3:
                affected = [0, 1, 2]
            for index in affected:
                after[start:end, index] = np.nan
        elif fault == "noise_burst":
            sigma = float(rng.uniform(.25, .65) * abs(magnitude))
            segment[finite] += rng.normal(0, sigma, end - start)[finite]
            params["noise_std"] = sigma
        elif fault == "clipping":
            past = original[max(0, start - 120):start, channel]
            finite_past = past[np.isfinite(past)]
            ceiling = bool(rng.integers(2))
            bound = float(np.quantile(finite_past, .35 if ceiling else .65)) if len(finite_past) else np.nan
            if np.isfinite(bound):
                segment[finite] = (np.minimum(segment[finite], bound) if ceiling else np.maximum(segment[finite], bound))
            params.update(bound=bound if np.isfinite(bound) else None, direction="upper" if ceiling else "lower")
        elif fault == "scale_error":
            factor = float(rng.uniform(.97, .985) if channel == 1 else rng.uniform(.65, .9))
            if rng.integers(2):
                factor = 1 / factor
            segment[finite] *= factor
            params["gain"] = factor
        elif fault == "sensor_swap":
            affected = [0, 2]
            joint = np.isfinite(original[start:end, 0]) & np.isfinite(original[start:end, 2])
            for first, second in ((0, 2), (2, 0)):
                after[start:end, first][joint] = original[start:end, second][joint]
        if fault not in {"dropout", "sensor_swap"}:
            after[start:end, channel] = segment
    equal = (after == original) | (np.isnan(after) & np.isnan(original))
    modified = ~equal
    # Never manufacture values where the original required input is missing.
    if np.any(np.isnan(original) & np.isfinite(after)):
        raise AssertionError("scenario attempted to fill an original missing value")
    values.loc[:, list(CHANNELS)] = after
    labels = np.full(len(values), "no_injection", dtype=object)
    labels[modified.any(axis=1)] = fault
    changed = np.flatnonzero(modified.any(axis=1))
    event = {"scenario_id": f"{source.window_id}|{fault}|{variant}", "version": VERSION,
             "source_window": source.window_id, "partition": source.split, "fault": fault,
             "channels": [CHANNELS[i] for i in affected], "parameters": params,
             "scheduled_start": baseline.timestamp.iloc[start].isoformat(),
             "scheduled_end_exclusive": baseline.timestamp.iloc[end].isoformat(),
             "first_modified": baseline.timestamp.iloc[changed[0]].isoformat() if len(changed) else None,
             "last_modified": baseline.timestamp.iloc[changed[-1]].isoformat() if len(changed) else None,
             "modified_rows": int(len(changed)), "modified_values": int(modified.sum()),
             "applied": bool(len(changed)), "source_file": source.source_file,
             "source_sha256": source.source_sha256,
             "label_semantics": "applied software modification; no_injection is not hardware-normal truth"}
    return Scenario(source, values, labels, modified, event)
