"""Per-channel robust statistical detector (the strong classical branch).

Operates on CLIMATOLOGY-RESIDUAL frames (station-month-hour seasonality
already removed), in units of robust sigma of training residuals:

    s_level   = |resid| / sigma            (bias, drift, spikes, clipping)
    s_step    = |diff|  / sigma_diff       (spike onsets, drops)
    stuckness = rolling-std == ~0 over observed reports (persistence evidence)
    noisiness = rolling-std >> train p99    (noise burst)
Per-timestamp scores are derived model outputs, not observations or fault
labels. Missing readings supply no anomaly evidence and are never filled.
Zero means no available detector evidence, not a reviewed fault-free label.
Scoring is causal: appending future reports cannot change an existing score.
Training history is not assumed to have independently confirmed fault status.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Detection channels: the three measured AWS parameters + physically derived
# quantities computed from them (SIH26073 constraint).
CORE = ["temperature_c", "pressure_hpa", "relative_humidity_pct",
        "dewpoint_c", "td_spread", "vpd_hpa", "press_tendency_3h"]

# Physical floors for scale parameters — keeps divisions sane on ultra-smooth
# channels (long drought RH plateaus, smooth reanalysis pressure, etc.)
MIN_SCALE = {"temperature_c": 0.3, "pressure_hpa": 0.4, "relative_humidity_pct": 3.0,
             "dewpoint_c": 0.3, "td_spread": 1.0, "es_hpa": 0.5,
             "vpd_hpa": 0.5, "press_tendency_3h": 0.3,
             "wind_speed_ms": 0.3, "precipitation_mm": 2.0, "solar_radiation_wm2": 20.0}


class RobustChannelDetector:
    def __init__(self, stuck_window: int = 8):
        self.stuck_window = stuck_window
        self.params: dict[str, dict[str, float]] = {}

    def fit(self, train_raw: pd.DataFrame, train_resid: pd.DataFrame,
            cols: list[str] | None = None) -> "RobustChannelDetector":
        """Fit on TRAIN data. Step/stuck/noise dynamics are learned on RAW values
        (a frozen sensor is frozen in raw units even under a moving diurnal
        cycle); level magnitudes are learned on climatology residuals."""
        cols = [c for c in (cols or CORE) if c in train_raw.columns]
        self.params = {}
        for c in cols:
            xr = train_raw[c].to_numpy(dtype=float)
            if not len(xr) or np.isfinite(xr).mean() < 0.5:
                continue
            xr = np.where(np.isfinite(xr), xr, np.nan)
            observed_raw = xr[np.isfinite(xr)]
            floor_c = MIN_SCALE.get(c, 0.25)
            d = np.abs(np.diff(xr))
            d = d[np.isfinite(d)]  # only observed consecutive differences
            sigma_d = float(max(1.4826 * np.median(d), 1e-3)) if len(d) else floor_c
            rstd = (pd.Series(xr).rolling(self.stuck_window, min_periods=self.stuck_window)
                    .std().dropna().to_numpy())
            med_raw = float(np.median(observed_raw))
            mad_raw = float(1.4826 * np.median(np.abs(observed_raw - med_raw)))
            std_hi = float(max(np.quantile(rstd, 0.995),
                               0.125 * mad_raw, sigma_d * 0.5, 1e-6)) if len(rstd) else 1.0
            std_lo = float(max(np.quantile(rstd, 0.005), 1e-9)) if len(rstd) else 1e-9

            if c in train_resid.columns:
                x = train_resid[c].to_numpy(dtype=float)
                x = np.where(np.isfinite(x), x, np.nan)
            else:
                x = xr
            observed_resid = x[np.isfinite(x)]
            if not len(observed_resid):
                continue
            med = float(np.median(observed_resid))
            sigma = float(max(1.4826 * np.median(np.abs(observed_resid - med)),
                              np.quantile(np.abs(observed_resid - med), 0.95) / 3.0, floor_c))
            stuck_thr = float(max(std_lo * 0.5, 1e-9))
            # weekly (168h) rolling-std of residuals — reference for gain faults
            r168 = (pd.Series(x).rolling(168, min_periods=24).std()
                    .dropna().to_numpy())
            rstd168_med = float(max(np.median(r168), 1e-6)) if len(r168) else sigma
            self.params[c] = dict(med=med, sigma=sigma,
                                  sigma_d=max(sigma_d, 0.25 * sigma, floor_c * 0.5),
                                  std_hi=std_hi, stuck_thr=stuck_thr,
                                  rstd168_med=rstd168_med)
        return self

    def _channel_score(self, raw: np.ndarray, resid: np.ndarray,
                       p: dict[str, float]) -> np.ndarray:
        if not len(raw):
            return np.zeros(0, dtype=float)
        ser = pd.Series(np.where(np.isfinite(raw), raw, np.nan))
        residual = pd.Series(np.where(np.isfinite(resid), resid, np.nan))
        s_level = (residual - p["med"]).abs().to_numpy() / p["sigma"]
        s_step = ser.diff().abs().to_numpy() / p["sigma_d"]

        # Require a complete observed window for persistence/noise evidence.
        # A gap breaks persistence instead of becoming a manufactured plateau.
        rstd = ser.rolling(self.stuck_window, min_periods=self.stuck_window).std().to_numpy()
        is_flat = pd.Series(rstd <= p["stuck_thr"])
        run = is_flat.astype(int).groupby((~is_flat).cumsum()).cumsum().to_numpy()
        stuckness = np.where(is_flat, 3.0 + 4.0 * np.clip(run / 48.0, 0, 1), 0.0)
        noisiness = np.clip(rstd / p["std_hi"] - 1.25, 0, None)

        # clipping/saturation: many points within eps of a stale rolling max,
        # while the series is not globally stuck
        rm = ser.rolling(24, min_periods=24).max()
        eps = 1e-9 + 1e-3 * p["sigma"]
        stale_max = rm.diff().abs().rolling(6, min_periods=6).max() <= eps
        pinned = ((rm - ser) <= eps) & stale_max & (rstd > p["stuck_thr"])
        clip_frac = pd.Series(pinned.astype(float)).rolling(24, min_periods=8).mean()
        clip_frac = clip_frac.fillna(0).to_numpy()
        clip_score = np.clip((clip_frac - 0.20) / 0.15, 0, None) * 2.0

        # gain/scale error: weekly rolling-std ratio vs the station's own norm
        r168 = residual.rolling(168, min_periods=24).std().to_numpy()
        observed = residual.notna()
        observed_run = observed.astype(int).groupby((~observed).cumsum()).cumsum().to_numpy()
        r168 = np.where(observed_run >= 24, r168, np.nan)
        ratio = r168 / p["rstd168_med"]
        ratio = np.maximum(ratio, 1.0 / np.maximum(ratio, 1e-9))
        scale_score = np.clip(ratio - 1.30, 0, None) * 1.5

        # Unavailable derived components make no contribution. This does not
        # create replacement observations or assign normal/fault labels.
        evidence = np.array([s_level, s_step, stuckness, noisiness,
                             clip_score, scale_score])
        evidence = np.where(np.isfinite(evidence), evidence, 0.0)
        score = np.maximum.reduce(evidence)
        score = np.where(ser.notna(), score, 0.0)
        return score

    def score(self, raw_df: pd.DataFrame, resid_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(raw_df))
        for c, p in self.params.items():
            if c in raw_df.columns:
                resid = resid_df[c] if c in resid_df.columns else raw_df[c]
                out = np.maximum(out, self._channel_score(
                    raw_df[c].to_numpy(dtype=float), resid.to_numpy(dtype=float), p))
        return out
