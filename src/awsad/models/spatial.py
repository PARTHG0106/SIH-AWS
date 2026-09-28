"""Spatial consistency detector (buddy check, titanlib-style) — implements the
PS example: 'an AWS reads 55C while neighbouring stations are normal'.

For each station-group and timestamp we compare the station's climatology-
residual against the robust median of its peers (nearest k stations, inverse-
distance weighted; the co-located Open-Meteo reanalysis twin of each real
station sits at distance ~0 and thus acts as a permanent independent referee).

Residual space removes gross climatology differences; residual disagreement is
normalised per (group, channel) against its clean-training spread.

References: metno/titanlib buddy_check & spatial consistency test (MET Norway),
NCBI/WMO guidance on spatial QC for observation networks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["temperature_c", "pressure_hpa", "relative_humidity_pct"]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


class SpatialConsistency:
    def __init__(self, station_meta: dict[str, dict], k_nearest: int = 4,
                 max_dist_km: float = 500.0, min_peers: int = 2):
        """station_meta: station_id -> {"lat","lon","elev"}. Groups are the
        station|source keys used everywhere else in awsad."""
        self.meta = station_meta
        self.k = k_nearest
        self.max_dist = max_dist_km
        self.min_peers = min_peers
        self.train_spread: dict[tuple[str, str], float] = {}
        self.channel_spread: dict[str, float] = {}

    # ------------------------------------------------------------------ peers
    def _peers(self, group: str, all_groups: list[str]) -> list[tuple[str, float]]:
        sid = group.split("|")[0]
        own = self.meta.get(sid)
        if not own:
            return []
        dists = []
        for g in all_groups:
            if g == group:
                continue
            gs = g.split("|")[0]
            # the co-located reanalysis twin of the same station: distance ~0
            if gs == sid:
                dists.append((g, 0.0))
                continue
            m = self.meta.get(gs)
            if not m:
                continue
            d = haversine_km(own["lat"], own["lon"], m["lat"], m["lon"])
            if d <= self.max_dist:
                dists.append((g, d))
        dists.sort(key=lambda t: t[1])
        return dists[: self.k]

    # ------------------------------------------------------------------ timeries pivot
    @staticmethod
    def _pivot(frames: dict[str, pd.DataFrame], channel: str) -> pd.DataFrame:
        series = {}
        for g, df in frames.items():
            if channel in df.columns and "timestamp" in df.columns:
                s = df.set_index("timestamp")[channel]
                series[g] = s
        if not series:
            return pd.DataFrame()
        return pd.concat(series, axis=1).sort_index()

    # ------------------------------------------------------------------ fit
    def fit(self, train_resid: dict[str, pd.DataFrame]) -> "SpatialConsistency":
        for c in CHANNELS:
            pv = self._pivot(train_resid, c)
            if pv.empty:
                continue
            fitted_spreads = []
            for g in train_resid:
                if g not in pv.columns:
                    continue
                peers = [p for p, _ in self._peers(g, list(train_resid))
                         if p in pv.columns]
                if len(peers) < self.min_peers:
                    continue
                med = pv[peers].median(axis=1)
                d = (pv[g] - med).abs()
                d = d[np.isfinite(d)]
                if len(d) < 100:
                    continue
                med = float(np.median(d))
                spread = float(max(1.4826 * np.median(np.abs(d - med)),
                                   np.quantile(d, 0.95) / 3.0, 1e-3))
                self.train_spread[(g, c)] = spread
                fitted_spreads.append(spread)
            if fitted_spreads:
                self.channel_spread[c] = float(np.median(fitted_spreads))
        return self

    # ------------------------------------------------------------------ score
    def score_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
        out = {g: np.zeros(len(f)) for g, f in frames.items()}
        for c in CHANNELS:
            pv = self._pivot(frames, c)
            if pv.empty:
                continue
            for g in pv.columns:
                peers = [p for p, _ in self._peers(g, list(frames)) if p in pv.columns]
                if len(peers) < self.min_peers:
                    continue
                med = pv[peers].median(axis=1)
                d = (pv[g] - med).abs()
                spread = self.train_spread.get((g, c))
                if spread is None:
                    # Held-out groups use a fallback learned from training groups.
                    # Never estimate a scale from the validation/test frame being
                    # scored; that would leak the anomalies into calibration.
                    spread = self.channel_spread.get(c)
                if spread is None:
                    continue
                score_c = (d / spread).to_numpy()
                score_c = np.where(np.isfinite(score_c), score_c, 0.0)
                # align pivot order back to frame order via timestamps
                ts = frames[g].set_index("timestamp").index
                dser = pd.Series(score_c, index=pv.index).reindex(ts).to_numpy()
                dser = np.where(np.isfinite(dser), dser, 0.0)
                out[g] = np.maximum(out[g], dser)
        return out
