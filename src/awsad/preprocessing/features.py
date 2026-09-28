"""Feature engineering for the classical (Isolation Forest) detector.

Design principle: models must generalise across stations, so all features are
*station-relative*: we first remove the station's own climatology (median by
day-of-year x hour, learned on TRAIN data only) and then normalise with robust
per-station statistics. A global model then learns climate-invariant dynamics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CORE_CHANNELS = [
    "temperature_c", "dewpoint_c", "relative_humidity_pct",
    "pressure_hpa", "wind_speed_ms", "wind_direction_deg",
    "precipitation_mm", "solar_radiation_wm2",
]

# --- PSI26073 compliance: detection runs on the three MEASURED AWS parameters
# plus physically-derived quantities computed from them (no other sensors).
MEASURED_CHANNELS = ["temperature_c", "pressure_hpa", "relative_humidity_pct"]
DERIVED_CHANNELS = ["dewpoint_c", "td_spread", "es_hpa", "vpd_hpa", "press_tendency_3h"]
DETECTION_CHANNELS = MEASURED_CHANNELS + DERIVED_CHANNELS


# ------------------------------------------------------------- physics derivations
def add_physics_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-derived channels from the three measured quantities.

    Dewpoint via Magnus inverse (Alduchov & Eskridge 1996 coefficients),
    saturation vapor pressure (Tetens), vapor-pressure deficit, and the 3-hour
    pressure tendency (WMO synoptic practice). All derived ONLY from the
    measured channels, keeping the pipeline PSI-conformant.
    """
    out = df.copy()
    a, b = 17.625, 243.04
    t = out["temperature_c"].to_numpy(dtype=float)
    rh = out["relative_humidity_pct"].to_numpy(dtype=float)
    p = out["pressure_hpa"].to_numpy(dtype=float)

    rh_c = np.clip(rh, 1e-3, 120.0)
    gamma = np.log(rh_c / 100.0) + a * t / (b + t)
    td = b * gamma / (a - gamma)
    td = np.clip(td, -60, 60)
    out["dewpoint_c"] = td
    out["td_spread"] = t - td

    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    out["es_hpa"] = es
    out["vpd_hpa"] = es * (1.0 - np.clip(rh, 0, 100) / 100.0)

    if "timestamp" in out.columns:
        s = pd.Series(p, index=out.index)
        # per-frame contiguous series (callers pass single-station/group frames)
        out["press_tendency_3h"] = s.diff(3).to_numpy()
    else:
        out["press_tendency_3h"] = np.nan
    return out


# --------------------------------------------------------------------------- RH
def dewpoint_to_rh(t_c: pd.Series, td_c: pd.Series) -> pd.Series:
    """Magnus formula (Alduchov & Eskridge 1996)."""
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c))
    return rh.clip(0, 100)


def impute_hourly(df: pd.DataFrame, cols: list[str], max_gap: int = 6) -> pd.DataFrame:
    """Hourly grid + short-gap interpolation; keeps a missingness mask column
    per channel. Uses resample-median so 30-minute airport reports are
    aggregated rather than silently dropped (asfreq would keep only hh:00)."""
    df = df.sort_values("timestamp")
    num = df.set_index("timestamp")[ [c for c in cols if c in df.columns] ]
    grid = num.resample("1h").median()
    # Carry provenance through resampling without allowing it into numeric
    # detector features.  This lets evaluation distinguish observed station
    # data from reanalysis references after the hourly grid is built.
    for meta in ("station_id", "source", "source_role", "rh_source",
                 "pressure_source"):
        if meta in df.columns:
            grid[meta] = df[meta].iloc[0]
    for c in [c for c in cols if c in grid.columns]:
        grid[f"{c}__was_missing"] = grid[c].isna().astype(np.float32)
        if max_gap > 0:
            grid[c] = grid[c].interpolate(limit=max_gap, limit_area="inside")
    # Carry real NOAA QC-anomaly labels through the hourly grid.  A flag is set
    # if ANY sub-hourly report in the hour was flagged (max), and the flagged
    # reading is aggregated by median so the actual anomalous value survives.
    indexed = df.set_index("timestamp")
    anom_cols = [c for c in df.columns if c.endswith("__qc_anomaly")]
    if anom_cols:
        anom = indexed[anom_cols].astype(float).resample("1h").max()
        for c in anom_cols:
            grid[c] = (anom[c].reindex(grid.index).fillna(0.0) > 0)
    value_cols = [c for c in df.columns if c.endswith("__qc_value")]
    if value_cols:
        vals = indexed[value_cols].resample("1h").median()
        for c in value_cols:
            grid[c] = vals[c].reindex(grid.index)
    return grid.reset_index().rename(columns={"index": "timestamp"})


def fit_climatology(train_df: pd.DataFrame, cols: list[str]) -> dict:
    """Fit (month, hour) median climatology on a CLEAN training frame."""
    month = train_df["timestamp"].dt.month
    hour = train_df["timestamp"].dt.hour
    clim: dict[str, dict] = {}
    for c in cols:
        if c not in train_df:
            continue
        keys = pd.DataFrame({"m": month, "h": hour, "v": train_df[c]})
        table = keys.groupby(["m", "h"])["v"].median()
        clim[c] = {"table": {f"{int(m)}_{int(h)}": float(v)
                             for (m, h), v in table.items()},
                   "fallback": float(train_df[c].median())}
    return clim


def apply_climatology(df: pd.DataFrame, cols: list[str], clim: dict) -> pd.DataFrame:
    """Subtract a previously-fitted climatology -> `__resid` columns."""
    out = df.copy()
    month = df["timestamp"].dt.month
    hour = df["timestamp"].dt.hour
    for c in cols:
        if c not in out or c not in clim:
            continue
        table = clim[c]["table"]
        mapped = [table.get(f"{m}_{h}", clim[c]["fallback"])
                  for m, h in zip(month, hour)]
        out[f"{c}__resid"] = out[c].to_numpy(dtype=float) - np.asarray(mapped)
    return out


def add_climatology_residuals(df: pd.DataFrame, cols: list[str],
                              train_mask: pd.Series
                              ) -> tuple[pd.DataFrame, dict]:
    """Back-compat helper: fit on train_mask rows, apply to whole frame."""
    clim = fit_climatology(df[train_mask], cols)
    return apply_climatology(df, cols, clim), clim


def add_window_features(df: pd.DataFrame, cols: list[str], use_aux: bool = True) -> pd.DataFrame:
    """Windowed dynamics: rolling stats, diffs, physical cross-channel features."""
    out = df.copy()
    for c in cols:
        base = f"{c}__resid" if f"{c}__resid" in out else c
        if base not in out:
            continue
        s = out[base]
        out[f"{c}__diff1"] = s.diff()
        out[f"{c}__roll_std_6h"] = s.rolling(6, min_periods=2).std()
        out[f"{c}__roll_mean_24h"] = s.rolling(24, min_periods=3).mean()
        out[f"{c}__roll_min_3h"] = s.rolling(3, min_periods=1).min()
        out[f"{c}__roll_max_3h"] = s.rolling(3, min_periods=1).max()
    if {"temperature_c", "dewpoint_c"} <= set(df.columns):
        out["td_spread_dyn"] = df["temperature_c"] - df["dewpoint_c"]
    if use_aux and "wind_speed_ms" in df.columns and "wind_direction_deg" in df.columns:
        rad = np.deg2rad(df["wind_direction_deg"].fillna(0))
        out["wind_u"] = -df["wind_speed_ms"].fillna(0) * np.sin(rad)
        out["wind_v"] = -df["wind_speed_ms"].fillna(0) * np.cos(rad)
    if "timestamp" in df.columns:
        h = df["timestamp"].dt.hour
        out["hour_sin"] = np.sin(2 * np.pi * h / 24)
        out["hour_cos"] = np.cos(2 * np.pi * h / 24)
        doy = df["timestamp"].dt.dayofyear
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


FEATURE_BLOCK_SUFFIXES = ("__resid", "__diff1", "__roll_std_6h", "__roll_mean_24h",
                          "__roll_min_3h", "__roll_max_3h", "__was_missing")
SCALAR_FEATURES = ["td_spread_dyn", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def feature_columns(df: pd.DataFrame, core_cols: list[str]) -> list[str]:
    cols = []
    for c in core_cols:
        for suf in FEATURE_BLOCK_SUFFIXES:
            name = f"{c}{suf}"
            if name in df.columns:
                cols.append(name)
    cols += [c for c in SCALAR_FEATURES if c in df.columns]
    return cols


def robust_scale_fit(df: pd.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
    params = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        finite = s[np.isfinite(s.to_numpy(dtype=float, na_value=np.nan))]
        # A source may legitimately omit a derived channel for an entire
        # station.  Persist an identity scale instead of allowing NaN to leak
        # into every downstream model and silently poison calibration.
        if finite.empty:
            params[c] = (0.0, 1.0)
            continue
        uniq = pd.unique(finite)
        if len(uniq) <= 2 and set(np.round(uniq.astype(float), 6)) <= {0.0, 1.0}:
            # binary indicator (e.g. __was_missing): keep 0/1 scale
            params[c] = (0.0, 1.0)
            continue
        med = float(finite.median())
        iqr = float(finite.quantile(0.75) - finite.quantile(0.25))
        std = float(finite.std())
        candidates = [x for x in (iqr, 0.5 * std) if np.isfinite(x)]
        scale = max(candidates + [1e-3])
        params[c] = (med, scale)
    return params


def robust_scale_apply(df: pd.DataFrame, params: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (med, iqr) in params.items():
        if c not in df:
            out[c] = np.nan
        else:
            scale = float(iqr) if np.isfinite(iqr) and abs(float(iqr)) > 1e-12 else 1.0
            center = float(med) if np.isfinite(med) else 0.0
            out[c] = (df[c] - center) / scale
    return out
