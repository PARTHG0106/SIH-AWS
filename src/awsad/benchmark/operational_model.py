"""Causal three-channel scenario-pattern model, separate from real fault truth.

Features use only current/past T/P/RH and timestamps for continuity. Rolling
statistics are derived features; no missing measurement is filled. The learned
probability concerns this synthetic scenario distribution, not hardware failure.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, softmax

from awsad.benchmark.scenarios import CHANNELS

FEATURE_VERSION = "three-channel-causal-pattern-v1"
CONTEXT_ROWS = 181
FLOORS = dict(zip(CHANNELS, (.1, .1, .3)))
BOUNDS = {"temperature_c": (-90., 60.), "pressure_hpa": (400., 1100.),
          "relative_humidity_pct": (0., 100.)}


def _features_contiguous(frame):
    output = {}
    normalized = {}
    for channel in CHANNELS:
        current = pd.to_numeric(frame[channel], errors="coerce").astype(float)
        current = current.where(np.isfinite(current))
        past = current.shift(1)
        low, high = BOUNDS[channel]
        output[channel + ":value"] = current
        output[channel + ":missing"] = current.isna().astype(float)
        output[channel + ":range_excess"] = np.maximum(low - current, np.maximum(current - high, 0))
        for lag in (1, 5, 15, 60, 120):
            output[f"{channel}:delta_{lag}"] = current - current.shift(lag)
        # Capped durations depend on at most 180 preceding records, matching the
        # bounded live inference history even after an arbitrarily long flatline.
        same = current.notna() & current.eq(past)
        run = same.groupby((~same).cumsum()).cumsum().clip(upper=180).astype(float)
        output[channel + ":repeat_minutes"] = run.where(current.notna())
        missing = current.isna()
        output[channel + ":missing_minutes"] = missing.groupby((~missing).cumsum()).cumsum().clip(upper=180)
        for window in (5, 30, 120):
            roll = past.rolling(window, min_periods=max(2, window // 2))
            mean, std = roll.mean(), roll.std(ddof=0)
            output[f"{channel}:past_mean_{window}"] = mean
            output[f"{channel}:past_std_{window}"] = std
            output[f"{channel}:deviation_{window}"] = current - mean
            output[f"{channel}:z_{window}"] = (current - mean) / std.clip(lower=FLOORS[channel])
            output[f"{channel}:coverage_{window}"] = past.rolling(window, min_periods=1).count() / window
        output[channel + ":short_long_variability"] = output[channel + ":past_std_5"] / output[channel + ":past_std_120"].clip(lower=FLOORS[channel])
        output[channel + ":mean_shift"] = output[channel + ":past_mean_5"] - output[channel + ":past_mean_120"]
        normalized[channel] = output[channel + ":z_120"]
    # Statistical relationships, not the tautological Td<=T check on derived Td.
    for i, left in enumerate(CHANNELS):
        for right in CHANNELS[i + 1:]:
            output[f"cross:{left}:{right}:deviation_product"] = normalized[left] * normalized[right]
            # Correlate changes to avoid cancellation from large pressure offsets.
            # Exact/near-constant changes have undefined correlation, explicitly
            # masked so bounded live windows match long batch windows numerically.
            a, b = output[left + ":value"].diff(), output[right + ":value"].diff()
            ra, rb = a.rolling(30, min_periods=15), b.rolling(30, min_periods=15)
            varied = (ra.max() - ra.min() > 1e-8) & (rb.max() - rb.min() > 1e-8)
            output[f"cross:{left}:{right}:correlation_30"] = ra.corr(b).clip(-1, 1).where(varied)
    return pd.DataFrame(output, index=frame.index).replace([np.inf, -np.inf], np.nan)


def causal_pattern_features(frame: pd.DataFrame) -> pd.DataFrame:
    if set(("timestamp", *CHANNELS)) - set(frame):
        raise ValueError("timestamp and the three measurement channels are required")
    timestamps = pd.to_datetime(frame.timestamp, utc=True)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and chronological within one stream")
    groups = timestamps.diff().ne(pd.Timedelta(minutes=1)).cumsum()
    pieces = [_features_contiguous(part) for _, part in frame.groupby(groups, sort=False)]
    return pd.concat(pieces).reindex(frame.index) if pieces else pd.DataFrame()


def temperature_probabilities(probabilities, temperature):
    logits = np.log(np.clip(np.asarray(probabilities, float), 1e-12, 1.))
    return softmax(logits / float(temperature), axis=1)


def fit_temperature(probabilities, target_indices):
    indices = np.asarray(target_indices, int)
    def objective(log_temperature):
        calibrated = temperature_probabilities(probabilities, np.exp(log_temperature))
        return -np.log(np.clip(calibrated[np.arange(len(indices)), indices], 1e-12, 1.)).mean()
    result = minimize_scalar(objective, bounds=(-2., 3.), method="bounded")
    return float(np.exp(result.x))


class OperationalPatternModel:
    def __init__(self, model, features, *, temperature=1., threshold=.5, metadata=None,
                 binary_calibration=None):
        self.model = model
        self.features = list(features)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.metadata = dict(metadata or {})
        self.binary_calibration = binary_calibration
        self.classes = np.asarray(model.classes_)
        self.background_index = int(np.flatnonzero(self.classes == "no_injection")[0])

    def predict_features(self, features):
        raw = self.model.predict_proba(features[self.features].to_numpy(np.float32))
        if self.binary_calibration is None:
            probabilities = temperature_probabilities(raw, self.temperature)
        else:
            raw_risk = np.clip(1 - raw[:, self.background_index], 1e-8, 1 - 1e-8)
            slope, intercept = self.binary_calibration
            risk = expit(slope * np.log(raw_risk / (1 - raw_risk)) + intercept)
            fault_mask = np.arange(len(self.classes)) != self.background_index
            conditional = raw[:, fault_mask]
            conditional = conditional / np.maximum(conditional.sum(axis=1, keepdims=True), 1e-12)
            conditional = temperature_probabilities(conditional, self.temperature)
            probabilities = np.zeros_like(raw)
            probabilities[:, self.background_index] = 1 - risk
            probabilities[:, fault_mask] = risk[:, None] * conditional
        risk = 1 - probabilities[:, self.background_index]
        fault_probabilities = probabilities.copy()
        fault_probabilities[:, self.background_index] = -1
        fault_indices = fault_probabilities.argmax(axis=1)
        return {"probabilities": probabilities, "scenario_score": risk,
                "is_candidate": risk >= self.threshold,
                "pattern": self.classes[probabilities.argmax(axis=1)],
                "pattern_confidence": probabilities.max(axis=1),
                "fault_pattern": self.classes[fault_indices],
                "fault_pattern_confidence": probabilities[np.arange(len(probabilities)), fault_indices]}

    def calibrate(self, features, labels):
        """Monotone binary calibration plus conditional type temperature.

        Separating these preserves detection ranking while calibrating the
        intervention probability and conditional type distribution separately.
        Only the held-out calibration partition belongs here.
        """
        raw = self.model.predict_proba(features[self.features].to_numpy(np.float32))
        labels = np.asarray(labels)
        changed = labels != "no_injection"
        risk = np.clip(1 - raw[:, self.background_index], 1e-8, 1 - 1e-8)
        logits = np.log(risk / (1 - risk))
        def loss(parameters):
            slope = np.exp(parameters[0])
            probabilities = np.clip(expit(slope * logits + parameters[1]), 1e-12, 1 - 1e-12)
            return float(-np.mean(changed * np.log(probabilities) + (~changed) * np.log(1 - probabilities)))
        fitted = minimize(loss, [0., 0.], method="L-BFGS-B", bounds=[(-3., 3.), (-12., 12.)])
        if not fitted.success:
            raise ValueError(f"probability calibration failed: {fitted.message}")
        self.binary_calibration = [float(np.exp(fitted.x[0])), float(fitted.x[1])]
        fault_mask = np.arange(len(self.classes)) != self.background_index
        conditional = raw[changed][:, fault_mask]
        conditional /= np.maximum(conditional.sum(axis=1, keepdims=True), 1e-12)
        fault_classes = self.classes[fault_mask]
        indices = {value: index for index, value in enumerate(fault_classes)}
        self.temperature = fit_temperature(conditional, [indices[value] for value in labels[changed]])
        return self

    def predict_frame(self, frame):
        return self.predict_features(causal_pattern_features(frame))

    def predict_one(self, history, scored=None):
        """Optional live hook; consumes at most 181 chronological raw rows."""
        frame = pd.DataFrame(history).tail(CONTEXT_ROWS)
        times = pd.to_datetime(frame.timestamp, utc=True)
        if len(frame) < CONTEXT_ROWS or not times.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
            return {"status": "warming_up", "is_candidate": False, "scenario_score": None,
                    "suggested_pattern": "unavailable", "pattern_confidence": None,
                    "required_contiguous_rows": CONTEXT_ROWS,
                    "confidence_semantics": "requires 180 minutes of contiguous past context",
                    "model_version": FEATURE_VERSION}
        result = self.predict_frame(frame)
        index = -1
        candidate = bool(result["is_candidate"][index])
        return {"status": "scored", "scenario_score": float(result["scenario_score"][index]),
                "is_candidate": bool(result["is_candidate"][index]),
                "suggested_pattern": str(result["fault_pattern"][index]) if candidate else "no_injection",
                "pattern_confidence": float(result["fault_pattern_confidence"][index]) if candidate
                    else float(result["probabilities"][index, self.background_index]),
                "threshold": self.threshold,
                "confidence_semantics": "calibrated on synthetic scenarios; not probability of real hardware failure",
                "model_version": FEATURE_VERSION}

    def save(self, path):
        joblib.dump({"model": self.model, "features": self.features, "temperature": self.temperature,
                     "threshold": self.threshold, "metadata": self.metadata,
                     "binary_calibration": self.binary_calibration,
                     "feature_version": FEATURE_VERSION}, path)

    @classmethod
    def load(cls, path):
        item = joblib.load(Path(path))
        if item.get("feature_version") != FEATURE_VERSION:
            raise ValueError("scenario model feature version mismatch")
        return cls(item["model"], item["features"], temperature=item["temperature"],
                   threshold=item["threshold"], metadata=item["metadata"],
                   binary_calibration=item.get("binary_calibration"))
