"""Synthetic sensor-fault injection engine with verifiable ground-truth labels.

Because no public labeled dataset of *weather-station sensor faults* exists
(verified during research), evaluation uses physically faithful synthetic
faults injected into real station data — the standard approach in the time-
series anomaly-detection literature (and the only way to get labels at scale).

Fault taxonomy (aligned with WMO CIMO guide & documented IMD AWS failure modes):
  spike        - single/short impulses of large magnitude
  drift        - gradual linear/exponential departure from truth
  bias         - sudden constant offset (miscalibration / siting change)
  stuck        - flatline; sensor reports constant value
  dropout      - missing data (NaN) or zero-fill telemetry gaps
  noise_burst  - elevated measurement noise for an interval
  clipping     - output saturates at instrument full-scale
  scale_error  - multiplicative gain error (wrong calibration curve)
  sensor_swap  - output mirrors another channel (wiring/config error)

Every injection appends rows to an events log (channel, type, start, end,
params) so models can be scored per fault CATEGORY as well — that's how we
can claim "no compromise on accuracy": every fault type is measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FAULT_WEIGHTS = {
    "spike": 0.16, "drift": 0.14, "bias": 0.14, "stuck": 0.16,
    "dropout": 0.12, "noise_burst": 0.10, "clipping": 0.08,
    "scale_error": 0.06, "sensor_swap": 0.04,
}
INJECTOR_VERSION = "sensor-swap-view-alias-fix-v3"

# Which channels a fault type plausibly affects — restricted to the three
# MEASURED AWS parameters required by SIH26073 (T, P, RH).
CHANNEL_FAULTS = {
    "spike": ["temperature_c", "pressure_hpa", "relative_humidity_pct"],
    "drift": ["temperature_c", "pressure_hpa", "relative_humidity_pct"],
    "bias": ["temperature_c", "pressure_hpa", "relative_humidity_pct"],
    "stuck": ["temperature_c", "pressure_hpa", "relative_humidity_pct"],
    "dropout": ["__any_measured__"],
    "noise_burst": ["temperature_c", "pressure_hpa", "relative_humidity_pct"],
    "clipping": ["relative_humidity_pct", "temperature_c"],
    "scale_error": ["temperature_c", "relative_humidity_pct"],
    "sensor_swap": [("temperature_c", "relative_humidity_pct")],
}


@dataclass
class InjectedEvent:
    fault: str
    channel: str
    start: pd.Timestamp
    end: pd.Timestamp
    params: dict = field(default_factory=dict)


def _robust_sigma(x: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 0:
        return 1.0
    return max(1.4826 * mad, 1e-3)


class AnomalyInjector:
    def __init__(self, rng: np.random.Generator | None = None,
                 target_fraction: float = 0.04,
                 min_len: int = 3, max_len: int = 168):
        self.rng = rng or np.random.default_rng(42)
        self.target_fraction = target_fraction
        self.min_len = min_len
        self.max_len = max_len

    # ------------------------------------------------------------------ faults
    # realistic per-fault duration ranges (hours) — a stuck sensor stays stuck,
    # a drift develops over days, a spike lasts minutes-hours
    SEG_LEN = {
        "spike": (1, 6), "drift": (48, 240), "bias": (12, 148),
        "stuck": (24, 168), "dropout": (4, 72), "noise_burst": (6, 72),
        "clipping": (12, 144), "scale_error": (24, 168), "sensor_swap": (24, 168),
    }

    def _inject(self, df: pd.DataFrame, fault: str,
                available_channels: list[str] | None = None) -> InjectedEvent | None:
        n = len(df)
        rng = self.rng
        lo, hi = self.SEG_LEN.get(fault, (self.min_len, self.max_len))
        hi = min(hi, max(lo + 1, n // 4))
        seg_len = int(rng.integers(lo, hi))
        start = int(rng.integers(0, max(1, n - seg_len)))
        end = start + seg_len

        available = set(available_channels or df.columns)
        channels = CHANNEL_FAULTS[fault][:]
        if channels == ["__any_measured__"]:
            channels = [c for c in ("temperature_c", "pressure_hpa",
                                    "relative_humidity_pct") if c in available]
        if fault == "sensor_swap":
            pairs = [(a, b) for a, b in channels
                     if a in available and b in available]
            if not pairs:
                return None
            ch_a, ch_b = pairs[0]
        else:
            channels = [c for c in channels if c in available]
            if not channels:
                return None
            ch_a = str(rng.choice(channels))

        x = df[ch_a].to_numpy(dtype=float, copy=True) if fault != "sensor_swap" else None
        if x is not None and not np.isfinite(x).any():
            return None
        # Avoid selecting a segment that starts on a missing value: such an
        # event would be labelled but could leave the signal unchanged.
        if x is not None:
            finite = np.flatnonzero(np.isfinite(x))
            if len(finite):
                if not np.isfinite(x[start]):
                    candidates = finite[(finite >= start) & (finite < end)]
                    if len(candidates):
                        start = int(candidates[0])
                    else:
                        start = int(finite[0])
                    # Moving the start to a finite observation must also move
                    # the end.  Otherwise the actual slice is shorter than
                    # ``seg_len`` and vector faults (drift/noise) cannot be
                    # broadcast onto it.
                    end = min(n, start + seg_len)
        actual_len = end - start
        if actual_len <= 0:
            return None
        sigma = _robust_sigma(x) if x is not None else 1.0
        params: dict = {}
        before = (x[start:end].copy() if x is not None else
                  df[ch_a].to_numpy(dtype=float, copy=True)[start:end])

        if fault == "spike":
            mag = rng.uniform(4, 10) * sigma * rng.choice([-1, 1])
            params = {"magnitude": float(mag)}
            if ch_a == "precipitation_mm":
                mag = abs(mag) * 3
            x[start:end] = x[start:end] + mag
        elif fault == "drift":
            total = rng.uniform(3, 8) * sigma * rng.choice([-1, 1])
            ramp = np.linspace(0, 1, actual_len) ** rng.uniform(0.7, 1.3)
            params = {"total_drift": float(total)}
            x[start:end] = x[start:end] + total * ramp
        elif fault == "bias":
            off = rng.uniform(2.5, 7) * sigma * rng.choice([-1, 1])
            params = {"offset": float(off)}
            x[start:end] = x[start:end] + off
        elif fault == "stuck":
            val = x[start]
            params = {"value": float(val)}
            x[start:end] = val
        elif fault == "dropout":
            mode = rng.choice(["nan", "zero"])
            params = {"mode": mode}
            x[start:end] = np.nan if mode == "nan" else 0.0
        elif fault == "noise_burst":
            mult = rng.uniform(4, 10)
            params = {"noise_multiplier": float(mult)}
            x[start:end] = x[start:end] + rng.normal(
                0, sigma * mult, actual_len
            )
        elif fault == "clipping":
            # Use a channel-aware plateau and verify that the selected window
            # actually changes.  A generic 90th percentile can be above every
            # value in a short segment, creating a false label.
            q = 0.75 if ch_a != "relative_humidity_pct" else 0.90
            clip_at = float(np.nanquantile(x, q))
            segment = x[start:end]
            if np.isfinite(segment).any() and np.nanmax(segment) <= clip_at:
                clip_at = float(np.nanmedian(segment))
            params = {"clip_at": clip_at}
            x[start:end] = np.minimum(segment, clip_at)
        elif fault == "scale_error":
            gain = rng.uniform(1.3, 2.0) ** rng.choice([-1, 1])
            mu = np.nanmean(x[start:end])
            params = {"gain": float(gain)}
            x[start:end] = mu + (x[start:end] - mu) * gain
        elif fault == "sensor_swap":
            b = df[ch_b].to_numpy(dtype=float)
            a = df[ch_a].to_numpy(dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            valid_idx = np.flatnonzero(valid)
            if len(valid_idx) < 2:
                return None
            if not valid[start:end].any():
                candidates = valid_idx[valid_idx >= start]
                start = int(candidates[0]) if len(candidates) else int(valid_idx[0])
                end = min(n, start + seg_len)
            actual_len = end - start
            if actual_len <= 0 or not valid[start:end].any():
                return None
            valid_a = a[valid]
            valid_b = b[valid]
            mean_a, mean_b = float(valid_a.mean()), float(valid_b.mean())
            scale_a = max(float(valid_a.std()), 1e-9)
            scale_b = max(float(valid_b.std()), 1e-9)
            rescaled = a.copy()
            rescaled[valid] = ((b[valid] - mean_b) / scale_b * scale_a
                               + mean_a)
            params = {"swapped_with": ch_b}
            segment_valid = valid[start:end]
            # ``a`` is a view over ch_a's column buffer, so the in-place
            # ``df.loc`` write below mutates it.  Snapshot the pre-write values
            # first; otherwise the change check compares the buffer against
            # itself and sensor_swap is silently dropped every time.
            original_segment = a[start:end].copy()
            replacement = np.where(segment_valid, rescaled[start:end],
                                   original_segment)
            changed = segment_valid & (original_segment != replacement)
            if not changed.any():
                return None
            df.loc[df.index[start:end], ch_a] = replacement
            return InjectedEvent(fault, ch_a, df["timestamp"].iat[start],
                                 df["timestamp"].iat[min(end - 1, n - 1)], params)

        changed = ~((before == x[start:end]) |
                    (np.isnan(before) & np.isnan(x[start:end])))
        if not changed.any():
            return None
        df.loc[df.index[start:end], ch_a] = x[start:end]
        return InjectedEvent(fault, ch_a, df["timestamp"].iat[start],
                             df["timestamp"].iat[min(end - 1, n - 1)], params)

    # ------------------------------------------------------------------ driver
    def inject(self, frame: pd.DataFrame, channel_cols: list[str],
               required_faults: list[str] | tuple[str, ...] | set[str] | None = None
               ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Inject faults into a single-station frame.

        Returns (corrupted_frame, point_labels, events_table).
        `point_labels` is 1 for every timestamp touched by any fault.
        """
        df = frame.copy()
        n = len(df)
        target_points = int(self.target_fraction * n)
        avg_len = (self.min_len + min(self.max_len, max(8, n // 10))) / 2
        n_events = max(1, int(target_points / avg_len))

        # Restrict the fault engine to the measured channels actually present
        # in this frame.  Keep labels initialized before the early-return
        # path, because partially populated station streams are legitimate.
        used_channel_cols = [c for c in channel_cols if c in df.columns]
        labels = np.zeros(n, dtype=int)
        supported = set(used_channel_cols)
        faults = [f for f in FAULT_WEIGHTS
                  if (f != "sensor_swap" or
                      {"temperature_c", "relative_humidity_pct"} <= supported)]
        faults = [f for f in faults if f != "__none__"]
        if not faults:
            return df, pd.Series(labels, index=df.index, name="label"), pd.DataFrame()
        probs = np.array([FAULT_WEIGHTS[f] for f in faults])
        probs = probs / probs.sum()

        events: list[InjectedEvent] = []
        work = df[["timestamp"] + used_channel_cols].copy()

        forced_faults = [str(f) for f in (required_faults or ())
                         if str(f) in faults]
        sampled_faults = forced_faults + [
            str(self.rng.choice(faults, p=probs)) for _ in range(n_events)
        ]
        for fault in sampled_faults:
            ev = self._inject(work, fault, used_channel_cols)
            if ev is None:
                continue
            events.append(ev)
            mask = (work["timestamp"] >= ev.start) & (work["timestamp"] <= ev.end)
            labels |= mask.to_numpy(dtype=int)

        df[used_channel_cols] = work[used_channel_cols]
        ev_df = pd.DataFrame(
            [{"fault": e.fault, "channel": e.channel, "start": e.start,
              "end": e.end, **{f"param_{k}": v for k, v in e.params.items()}}
             for e in events])
        return df, pd.Series(labels, index=df.index, name="label"), ev_df
