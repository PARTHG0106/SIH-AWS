"""Causal multi-signal detection on unchanged native station measurements.

Threshold exceedances are candidates for review, never hardware-fault labels.
QC is used for fitting/calibration eligibility only, not as a model predictor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

CHANNELS = ("temperature_c", "pressure_hpa", "relative_humidity_pct")
SIGNALS = ("forecast_residual", "abrupt_change", "flatline_minutes", "hour_residual", "sustained_deviation")
BASE_COLUMNS = ["observation_id", "timestamp", "station_id", "source", *CHANNELS,
                *(c + "__qc_accepted" for c in CHANNELS)]


@dataclass(frozen=True)
class MinuteConfig:
    train_end: str = "2024-07-01T00:00:00Z"
    selection_end: str = "2024-09-01T00:00:00Z"
    calibration_end: str = "2024-11-01T00:00:00Z"
    holdout_station_ids: tuple[str, ...] = ("gwn",)
    horizons: tuple[int, ...] = (1, 60)
    context_minutes: tuple[int, ...] = (0, 5, 15, 60)
    sustained_minutes: int = 30
    reference_alert_budget: float = 0.005
    min_calibration_rows: int = 1000
    sample_per_month: int = 1800
    max_iter: int = 100
    random_seed: int = 42
    max_cpu_threads: int = 4

    def validate(self):
        dates = [pd.Timestamp(x) for x in (self.train_end, self.selection_end, self.calibration_end)]
        if any(t.tzinfo is None for t in dates) or not dates[0] < dates[1] < dates[2]:
            raise ValueError("strictly ordered timezone-aware split boundaries are required")
        if self.horizons != (1, 60) or self.context_minutes != (0, 5, 15, 60):
            raise ValueError("this detector's reviewed feature schema uses horizons 1/60 and lags 0/5/15/60")
        if not 0 < self.reference_alert_budget < 1 or self.sustained_minutes < 2:
            raise ValueError("invalid reference budget or sustained window")
        if min(self.min_calibration_rows, self.sample_per_month, self.max_iter, self.max_cpu_threads) < 1:
            raise ValueError("sample and model budgets must be positive")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def validate_frame(frame):
    if set(BASE_COLUMNS) - set(frame):
        raise ValueError("native identity, channels and provider QC columns required")
    if frame.empty:
        raise ValueError("empty native frame")
    ts = pd.to_datetime(frame.timestamp)
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise ValueError("native timestamps must be timezone aware")
    if frame[["source", "station_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("one station|source group per frame is required")
    if ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError("native timestamps must be unique and chronological")
    if frame.observation_id.isna().any() or frame.observation_id.duplicated().any():
        raise ValueError("unique original observation IDs are required")
    if "label" in frame and frame.label.notna().any():
        raise ValueError("this native observation path requires unknown fault labels")


def split_names(frame, cfg):
    t = frame.timestamp
    return np.select([t < pd.Timestamp(cfg.train_end), t < pd.Timestamp(cfg.selection_end),
                      t < pd.Timestamp(cfg.calibration_end)],
                     ["train", "selection", "calibration"], default="reserved")


def causal_features(frame, horizon, cfg):
    """Exact past-time lookup; no gap filling or QC/model-derived observations."""
    validate_frame(frame)
    lookup = frame.set_index("timestamp")
    blocks, quality, ids = [], [], {}
    for lag in cfg.context_minutes:
        past = lookup.reindex(frame.timestamp - pd.Timedelta(minutes=horizon + lag))
        blocks.append(past[list(CHANNELS)].to_numpy(float, na_value=np.nan))
        quality.append(past[[c + "__qc_accepted" for c in CHANNELS]].fillna(False).to_numpy(bool).all(axis=1))
        ids[str(horizon + lag)] = past.observation_id.to_numpy()
    x = np.column_stack(blocks + [blocks[0] - b for b in blocks[1:]])
    # All predictors come from the three observed channels. No station ID, QC,
    # calendar covariates or external/reanalysis measurements enter the model.
    return {"X": x, "complete": np.isfinite(x).all(axis=1),
            "accepted": np.column_stack(quality).all(axis=1),
            "baseline": blocks[0], "context_ids": ids}


def flatline_durations(frame, state=None, *, return_quality=False):
    """Elapsed exact-repeat minutes; gaps/missing values reset, never backdate."""
    state = dict(state or {})
    result, quality = {}, {}
    times = pd.DatetimeIndex(frame.timestamp)
    for c in CHANNELS:
        values = frame[c].to_numpy(float, na_value=np.nan)
        previous = state.get(c)
        equal = np.zeros(len(values), bool)
        equal[1:] = ((np.diff(times.as_unit("ns").asi8) == 60_000_000_000)
                     & np.isfinite(values[1:]) & (values[1:] == values[:-1]))
        offset = 0
        if previous and times[0] - pd.Timestamp(previous[0]) == pd.Timedelta(minutes=1):
            if np.isfinite(values[0]) and values[0] == previous[1]:
                equal[0] = True
                offset = previous[2] + 1
        starts = np.maximum.accumulate(np.where(~equal, np.arange(len(values)), 0))
        duration = (np.arange(len(values)) - starts).astype(float)
        if equal[0]:
            first_reset = np.flatnonzero(~equal)
            duration[:first_reset[0] if len(first_reset) else len(duration)] += offset
        duration[~np.isfinite(values)] = np.nan
        result[c] = duration
        accepted = frame[c + "__qc_accepted"].fillna(False).to_numpy(bool) & np.isfinite(values)
        bad = np.cumsum(~accepted)
        prior_bad = np.where(starts > 0, bad[np.maximum(starts - 1, 0)], 0)
        good = bad == prior_bad
        if equal[0] and (len(previous) < 4 or not previous[3]):
            first_reset = np.flatnonzero(~equal)
            good[:first_reset[0] if len(first_reset) else len(good)] = False
        quality[c] = good
        state[c] = (str(times[-1]), float(values[-1]), float(duration[-1]), bool(good[-1]))
    return (result, state, quality) if return_quality else (result, state)


def fit_forecasters(samples, cfg):
    cfg.validate()
    models, selection = {}, {}
    with threadpool_limits(limits=cfg.max_cpu_threads):
        for h in cfg.horizons:
            for i, c in enumerate(CHANNELS):
                key = f"{h}:{c}"
                fit, val = samples["train"][h], samples["selection"][h]
                m = fit["accepted"] & np.isfinite(fit["y"][:, i]) & fit["target_accepted"][:, i]
                v = val["accepted"] & np.isfinite(val["y"][:, i]) & val["target_accepted"][:, i]
                if m.sum() < 200 or v.sum() < 100:
                    raise ValueError(f"insufficient original fit/selection examples for {key}")
                model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=cfg.max_iter,
                    max_leaf_nodes=15, min_samples_leaf=30, learning_rate=.05,
                    l2_regularization=3., early_stopping=False, random_state=cfg.random_seed)
                model.fit(fit["X"][m], fit["y"][m, i] - fit["baseline"][m, i])
                base = val["baseline"][v, i]
                predicted = base + model.predict(val["X"][v])
                baseline_mae = float(np.abs(val["y"][v, i] - base).mean())
                model_mae = float(np.abs(val["y"][v, i] - predicted).mean())
                selected = "gradient_boosted_delta" if model_mae < baseline_mae else "persistence"
                models[key] = model if selected == "gradient_boosted_delta" else None
                selection[key] = {"selected": selected, "fit_rows": int(m.sum()), "selection_rows": int(v.sum()),
                    "persistence_mae": baseline_mae, "gradient_boosted_delta_mae": model_mae,
                    "metric_semantics": "forecasting MAE on provider-accepted observations; not fault accuracy"}
    return models, selection


def compute_signals(frame, models, cfg):
    """Score current observations against past-only predictions/context."""
    predictions, features = {}, {}
    with threadpool_limits(limits=cfg.max_cpu_threads):
        for h in cfg.horizons:
            f = features[h] = causal_features(frame, h, cfg)
            pred = np.full((len(frame), len(CHANNELS)), np.nan)
            ok = f["complete"]
            for i, c in enumerate(CHANNELS):
                model = models[f"{h}:{c}"]
                pred[ok, i] = f["baseline"][ok, i]
                if model is not None and ok.any():
                    pred[ok, i] += model.predict(f["X"][ok])
            predictions[h] = pred
    flat, _, flat_quality = flatline_durations(frame, return_quality=True)
    gap = frame.timestamp.diff().ne(pd.Timedelta(minutes=1))
    runs = gap.cumsum()
    result, eligibility = {}, {}
    target_qc = frame[[c + "__qc_accepted" for c in CHANNELS]].fillna(False).to_numpy(bool)
    for i, c in enumerate(CHANNELS):
        y = frame[c].to_numpy(float, na_value=np.nan)
        residual1, residual60 = y - predictions[1][:, i], y - predictions[60][:, i]
        signed = pd.Series(residual60, index=frame.index)
        sustained = signed.groupby(runs).transform(lambda s: s.rolling(cfg.sustained_minutes, min_periods=cfg.sustained_minutes).mean()).abs().to_numpy()
        accepted = pd.Series(target_qc[:, i] & features[60]["accepted"], index=frame.index)
        sustained_qc = accepted.groupby(runs).transform(lambda s: s.rolling(cfg.sustained_minutes, min_periods=cfg.sustained_minutes).sum()).eq(cfg.sustained_minutes).to_numpy()
        prior = frame[c].shift(1).to_numpy(float, na_value=np.nan)
        abrupt = np.abs(y - prior)
        abrupt[gap.to_numpy()] = np.nan
        result[c] = {"forecast_residual": np.abs(residual1), "abrupt_change": abrupt,
                     "flatline_minutes": flat[c], "hour_residual": np.abs(residual60),
                     "sustained_deviation": sustained}
        prior_qc = np.r_[False, target_qc[:-1, i]] & ~gap.to_numpy()
        eligibility[c] = {"forecast_residual": target_qc[:, i] & features[1]["accepted"],
                          "abrupt_change": target_qc[:, i] & prior_qc,
                          "flatline_minutes": flat_quality[c],
                          "hour_residual": target_qc[:, i] & features[60]["accepted"],
                          "sustained_deviation": sustained_qc}
    return result, eligibility, predictions, features


def fit_thresholds(calibration, cfg):
    """Frozen empirical quantiles, including a pooled unseen-group fallback.

    Budget divided across channels/signals controls calibration reference tails;
    it is not a guaranteed false-alarm rate and is not fault probability.
    """
    if "pooled_seen_groups" not in calibration:
        raise ValueError("no eligible seen-group calibration observations in the selected view")
    q = 1 - cfg.reference_alert_budget / (len(CHANNELS) * len(SIGNALS))
    thresholds = {}
    for group, channels in calibration.items():
        thresholds[group] = {}
        for c in CHANNELS:
            thresholds[group][c] = {}
            for signal in SIGNALS:
                x = np.concatenate(channels[c][signal]) if channels[c][signal] else np.array([])
                x = x[np.isfinite(x)]
                if len(x) < cfg.min_calibration_rows:
                    raise ValueError(f"insufficient calibration for {group}/{c}/{signal}")
                thresholds[group][c][signal] = {"threshold": float(np.quantile(x, q, method="higher")),
                    "reference_rows": len(x), "quantile": q, "comparison": "strictly_greater"}
    return thresholds


def apply_thresholds(frame, signals, predictions, thresholds, cfg):
    out = frame.copy()
    group = f"{frame.station_id.iloc[0]}|{frame.source.iloc[0]}"
    threshold_group = group if group in thresholds else "pooled_seen_groups"
    out["group"] = group
    out["split"] = split_names(frame, cfg)
    out["threshold_group"] = threshold_group
    out["station_partition"] = "holdout" if frame.station_id.iloc[0] in cfg.holdout_station_ids else "seen"
    any_alert, any_scored = np.zeros(len(out), bool), np.zeros(len(out), bool)
    overall = np.full(len(out), np.nan)
    reasons = np.full(len(out), "", dtype=object)
    for i, c in enumerate(CHANNELS):
        channel_alert, channel_scored = np.zeros(len(out), bool), np.zeros(len(out), bool)
        channel_score = np.full(len(out), np.nan)
        why = np.full(len(out), "", dtype=object)
        for name in SIGNALS:
            value = signals[c][name]
            threshold = thresholds[threshold_group][c][name]["threshold"]
            finite = np.isfinite(value)
            flag = finite & (value > threshold)
            # Ratio expresses exceedance; no confidence/probability mapping.
            ratio = value / threshold if threshold > 0 else np.where(value > 0, 2., 0.)
            ratio[~finite] = np.nan
            channel_score = np.fmax(channel_score, ratio)
            channel_alert |= flag
            channel_scored |= finite
            why[flag] = [s + ("|" if s else "") + name for s in why[flag]]
            reasons[flag] = [s + ("|" if s else "") + c + ":" + name for s in reasons[flag]]
            out[c + "__" + name] = value
        out[c + "__prediction"] = predictions[1][:, i]
        out[c + "__prediction_60m"] = predictions[60][:, i]
        out[c + "__score"] = channel_score
        out[c + "__reason_codes"] = why
        out[c + "__alert"] = pd.array(np.where(channel_scored, channel_alert, None), dtype="boolean")
        any_alert |= channel_alert
        any_scored |= channel_scored
        overall = np.fmax(overall, channel_score)
    out["anomaly_score"] = overall
    out["is_candidate"] = any_alert
    out["reason_codes"] = reasons
    out["scoring_status"] = np.where(any_scored, "scored_available_signals", "missing_observation")
    out["hardware_fault_status"] = "unknown"
    return out


def config_fingerprint(cfg):
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
