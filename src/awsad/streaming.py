"""Real-time streaming detector — O(1) state per observation.

SIH26073 requires real-time detection per AWS observation. This detector keeps
only bounded rolling buffers; per new row it produces score + flag +
explanation within microseconds (statistical fast path) while full-batch
models (IF/AE/forecaster) can refit/re-score periodically upstream.

Climatology lookup, robust-z, step limits, flatline run detection and the
    station's fast-path threshold come from the trained artifacts. The streaming
    path intentionally uses only physics QC and robust channel statistics; it is
    calibrated separately from the full batch ensemble.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from .models.ensemble import WeightedEnsemble
from .preprocessing.qc_rules import PHYSICAL_LIMITS, STEP_LIMITS

STREAM_CHANNELS = ["temperature_c", "pressure_hpa", "relative_humidity_pct"]


class StreamingDetector:
    def __init__(self, group: str, artifacts_dir: str | Path,
                 window: int = 24):
        art = Path(artifacts_dir)
        self.group = group
        self.window = window
        self.ens = WeightedEnsemble.load(str(art / "ensemble.json"))
        stream_path = art / "stream_thresholds.json"
        if stream_path.exists():
            stream_cfg = json.loads(stream_path.read_text())
            self.stream_weights = stream_cfg.get("components", {})
            self.threshold = stream_cfg.get("per_group", {}).get(
                group, stream_cfg.get("per_group_pot_fallback", {}).get(
                    group, stream_cfg.get("pot_fallback", self.ens.threshold_for(group))))
        else:
            self.stream_weights = {k: self.ens.weights.get(k, 0.0)
                                   for k in ("qc", "zscore")}
            self.threshold = self.ens.threshold_for(group)
        feature_meta = json.loads((art / "feature_meta.json").read_text())
        per_core = dict(feature_meta["groups"].get(group, {}))
        self.climatology = (per_core.get("climatology") or {})
        rcd = joblib.load(art / "robust_channel.joblib")
        self.params = rcd[group].params

        # rolling state
        state_channels = set(STREAM_CHANNELS) | set(self.params)
        self.buf = {c: deque(maxlen=window) for c in state_channels}
        self.run_len = {c: 0 for c in state_channels}
        self.prev = {c: None for c in state_channels}
        self.ewma_val = {c: 0.0 for c in state_channels}
        self.ewma_beta = 0.5       # halflife ~1.4h
        self.n = 0

    # ------------------------------------------------------------------ core
    def update(self, obs: dict) -> dict:
        """One observation -> realtime decision dict."""
        t0 = time.perf_counter_ns()
        self.n += 1
        ts = pd_ts = obs.get("timestamp")
        month = getattr(pd_ts, "month", 1); hour = getattr(pd_ts, "hour", 0)

        values = dict(obs)
        t = values.get("temperature_c")
        rh = values.get("relative_humidity_pct")
        p_now = values.get("pressure_hpa")
        if t is not None and rh is not None and np.isfinite(t) and np.isfinite(rh):
            a, b = 17.625, 243.04
            rh_c = float(np.clip(rh, 1e-3, 120.0))
            gamma = np.log(rh_c / 100.0) + a * t / (b + t)
            td = float(np.clip(b * gamma / (a - gamma), -60, 60))
            es = float(6.112 * np.exp(17.67 * t / (t + 243.5)))
            values.update({"dewpoint_c": td, "td_spread": t - td,
                           "vpd_hpa": es * (1.0 - np.clip(rh, 0, 100) / 100.0)})
        p_hist = list(self.buf.get("pressure_hpa", ()))
        if (p_now is not None and np.isfinite(p_now) and len(p_hist) >= 3
                and np.isfinite(p_hist[-3])):
            values["press_tendency_3h"] = p_now - p_hist[-3]

        chan_scores, qc_hits = {}, {}
        for c, p in self.params.items():
            if c not in STREAM_CHANNELS and c not in values:
                continue
            v = values.get(c)
            if v is None or not np.isfinite(v):
                chan_scores[c] = 4.0
                qc_hits.setdefault("missing", 0)
                qc_hits["missing"] += 1
                self.buf[c].append(np.nan)
                self.prev[c] = None
                continue
            lo, hi = PHYSICAL_LIMITS.get(c, (-1e9, 1e9))
            if v < lo or v > hi:
                qc_hits["range"] = qc_hits.get("range", 0) + 1
            clim = self.climatology.get(c, {})
            expected = clim.get("table", {}).get(f"{month}_{hour}",
                                                 clim.get("fallback", v))
            resid = v - expected
            s_level = abs(resid) / p["sigma"]
            s_step = 0.0
            if self.prev[c] is not None:
                dv = abs(v - self.prev[c])
                s_step = dv / p["sigma_d"]
                lim = STEP_LIMITS.get(c)
                if lim and dv > lim:
                    qc_hits["step"] = qc_hits.get("step", 0) + 1
                self.run_len[c] = self.run_len[c] + 1 if abs(v - self.prev[c]) <= 1e-9 else 0
            buf = self.buf[c]
            stuckness = 0.0
            if buf is not None:
                buf.append(v)
                if len(buf) >= 8 and float(np.std(list(buf)[-8:])) <= p["stuck_thr"]:
                    stuckness = 3.0
                    qc_hits["flatline"] = qc_hits.get("flatline", 0) + 1
            s_raw = max(s_level, s_step, stuckness)
            # EWMA-sustain (mirrors batch _ewma_max shaping)
            self.ewma_val[c] = self.ewma_beta * s_raw + (1 - self.ewma_beta) * self.ewma_val[c]
            chan_scores[c] = max(s_raw, self.ewma_val[c])
            self.prev[c] = v

        zs = max(chan_scores.values()) if chan_scores else 0.0
        qc = min(1.0, sum(qc_hits.values()) / 2.0)
        # ensemble fusion with this group's trained norms
        norms = self.ens.per_group_norms.get(self.group, self.ens.norms_)
        comp_map = {"zscore": zs, "qc": qc}
        tot, wsum = 0.0, 0.0
        for k, w in self.stream_weights.items():
            if k in comp_map and k in norms:
                m, h = norms[k]
                tot += w * max(0.0, (comp_map[k] - m) / max(h - m, 1e-9))
                wsum += w
        score = tot / max(wsum, 1e-9)
        conf = float(1.0 / (1.0 + np.exp(-2.0 * (score / max(self.threshold, 1e-9) - 1.0))))
        hard = bool(qc_hits.get("range") or qc_hits.get("flatline"))
        if hard:
            conf = max(conf, 0.98)
        top_ch = max(chan_scores, key=chan_scores.get) if chan_scores else None
        return {
            "timestamp": str(ts), "group": self.group, "score": float(score),
            "threshold": float(self.threshold),
            "is_anomaly": bool(score >= self.threshold or hard),
            "confidence": conf, "top_channel": top_ch, "qc_flags": qc_hits,
            "channel_scores": chan_scores,
            "latency_us": (time.perf_counter_ns() - t0) / 1000.0,
        }


def benchmark_latency(detector: StreamingDetector, rows: list[dict]) -> dict:
    lat = []
    flags = 0
    for r in rows:
        out = detector.update(r)
        lat.append(out["latency_us"])
        flags += int(out["is_anomaly"])
    lat = np.array(lat[100:])  # skip warm-up
    return {"n": len(lat), "mean_us": float(lat.mean()),
            "p95_us": float(np.quantile(lat, 0.95)),
            "throughput_obs_per_s": float(1e6 / max(lat.mean(), 1e-9)),
            "flags": flags}
