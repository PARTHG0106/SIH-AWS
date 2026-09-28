"""Causal quality-control heuristics for a single station's observations.

Flags and scores are derived model outputs, not observed fault labels or
confirmed hardware diagnoses. Persistence counts reports, not inferred hours.
Appending future reports must never change an already issued flag. The input
frame, including missing observations, is left unchanged.

The legacy range/step limits assume hourly Indian weather and sea-level
pressure; source-specific validation is required before using another view.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (min, max) physically plausible bounds, deliberately generous for India
PHYSICAL_LIMITS = {
    "temperature_c": (-35.0, 55.0),          # Leh winter .. extreme heatwave
    "dewpoint_c": (-45.0, 35.0),
    "relative_humidity_pct": (0.0, 100.0),
    "pressure_hpa": (850.0, 1090.0),         # msl pressure; Indian range 950-1050
    "wind_speed_ms": (0.0, 75.0),            # cyclonic gusts possible
    "wind_direction_deg": (0.0, 360.0),
    "precipitation_mm": (0.0, 350.0),        # per-hour world-record margin
    "solar_radiation_wm2": (0.0, 1450.0),
}

# Max physically-consistent hourly rate of change
STEP_LIMITS = {
    "temperature_c": 12.0,
    "dewpoint_c": 12.0,
    "pressure_hpa": 8.0,
    "wind_speed_ms": 30.0,
    "relative_humidity_pct": 50.0,
}

SPIKE_Z = 8.0           # |second-difference| relative to robust sigma
FLATLINE_MIN_LEN = 24   # exact-repeat reports => review (short plateaus are
                        # common for quantized RH/temperature in calm weather)
FLATLINE_EPS = 1e-6

# Repeated exact-zero observations are a heuristic for review. Their values
# alone cannot establish that a logger filled a gap or that a sensor failed.
DROPOUT_MIN_LEN = 3
DROPOUT_CHANNELS = ("temperature_c", "dewpoint_c", "pressure_hpa",
                    "relative_humidity_pct")


def physical_range_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col in df:
            flags[f"range_{col}"] = (df[col] < lo) | (df[col] > hi)
    return flags


def internal_consistency_flags(df: pd.DataFrame) -> pd.Series:
    """Dewpoint must not exceed air temperature (0.5C tolerance for rounding)."""
    out = pd.Series(False, index=df.index)
    if {"dewpoint_c", "temperature_c"} <= set(df.columns):
        out |= df["dewpoint_c"] > df["temperature_c"] + 0.5
    return out


def step_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    for col, lim in STEP_LIMITS.items():
        if col in df:
            flags[f"step_{col}"] = df[col].diff().abs() > lim
    return flags


def spike_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Second difference against a scale learned only from earlier reports."""
    flags = pd.DataFrame(index=df.index)
    for col in PHYSICAL_LIMITS:
        if col not in df:
            continue
        x = df[col]
        d2 = (x - 2 * x.shift(1) + x.shift(2)).abs()
        past_steps = x.diff().abs().shift(1)
        # An expanding historical median supports startup without taking the
        # standard deviation of the complete frame (which sees future spikes).
        # These are derived scales; absent observations remain absent.
        scale = (past_steps.rolling(72, min_periods=12).median()
                 .fillna(past_steps.expanding(min_periods=2).median())
                 .clip(lower=1e-3))
        flags[f"spike_{col}"] = d2 > SPIKE_Z * scale
    return flags


def flatline_flags(df: pd.DataFrame, min_len: int = FLATLINE_MIN_LEN) -> pd.DataFrame:
    """Flag from the first report with sufficient *past* repeat evidence."""
    flags = pd.DataFrame(index=df.index)
    for col in ("temperature_c", "dewpoint_c", "pressure_hpa",
                "relative_humidity_pct", "wind_speed_ms"):
        if col not in df:
            continue
        x = df[col]
        valid = pd.Series(np.isfinite(x), index=df.index)
        same = valid & valid.shift(1, fill_value=False) & (x.diff().abs() <= FLATLINE_EPS)
        grp = (~same).cumsum()
        runlen = valid.astype(int).groupby(grp).cumsum()
        flags[f"flatline_{col}"] = valid & (runlen >= min_len)
    return flags


def dropout_flags(df: pd.DataFrame, min_len: int = DROPOUT_MIN_LEN) -> pd.DataFrame:
    """Causal exact-zero persistence heuristic; the name is a legacy API.

An observed zero run is not proof of a telemetry dropout. Missing reports do
not contribute to this run, and the first ``min_len - 1`` zeros are not flagged.
"""
    flags = pd.DataFrame(index=df.index)
    for col in DROPOUT_CHANNELS:
        if col not in df:
            continue
        x = df[col]
        is_zero = (x.abs() < FLATLINE_EPS) & x.notna()
        grp = (is_zero != is_zero.shift()).cumsum()
        runlen = is_zero.astype(int).groupby(grp).cumsum()
        flags[f"dropout_{col}"] = is_zero & (runlen >= min_len)
    return flags


def compute_qc_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """All QC flags for a single-station frame (must be time-sorted)."""
    parts = [
        physical_range_flags(frame),
        internal_consistency_flags(frame).rename("consistency_td_gt_t"),
        step_flags(frame),
        spike_flags(frame),
        flatline_flags(frame),
        dropout_flags(frame),
    ]
    flags = pd.concat(parts, axis=1).fillna(False).astype(bool)
    flags["qc_flag"] = flags.any(axis=1)
    return flags


def hard_flag(flags: pd.DataFrame) -> pd.Series:
    """Legacy deterministic alert override; flags remain heuristic evidence."""
    cols = [c for c in flags.columns
            if c.startswith(("range_", "flatline_", "consistency_", "dropout_"))]
    if not cols:
        return pd.Series(False, index=flags.index)
    return flags[cols].any(axis=1)


def qc_score(flags: pd.DataFrame) -> pd.Series:
    """0..1 severity score: fraction-weighted, flatline/range/dropout heavier."""
    if flags.empty:
        return pd.Series(dtype=float)
    cols = [c for c in flags.columns if c != "qc_flag"]
    if not cols:
        return pd.Series(0.0, index=flags.index)
    w = np.ones(len(cols))
    for i, c in enumerate(cols):
        if c.startswith(("range_", "flatline_", "consistency_", "dropout_")):
            w[i] = 2.0
    raw = flags[cols].to_numpy(dtype=float) @ w
    return pd.Series(raw / w.sum(), index=flags.index, name="qc_score")
