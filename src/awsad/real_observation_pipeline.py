"""Causal forecasting experiment on original three-channel observations.

The caller owns provider verification and field-level provenance. This module
does not certify a source or infer fault truth. It never fills an observation,
and provider-accepted rows are not described as fault-free. Forecast residuals
are model outputs, with thresholds calibrated without fault labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import ThreadpoolController

CHANNELS = ("temperature_c", "pressure_hpa", "relative_humidity_pct")
KEYS = ("source", "station_id", "timestamp")


@dataclass(frozen=True)
class RealObservationConfig:
    horizon_minutes: tuple[int, ...] = (60,)
    context_lags_minutes: tuple[int, ...] = (0, 60, 180, 1440)
    seasonal_period_minutes: int = 1440
    thinning_minutes: int | None = 60
    train_end: str | None = None
    validation_end: str | None = None
    test_end: str | None = None
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    train_scale_fraction: float = 0.15
    validation_selection_fraction: float = 0.50
    holdout_station_ids: tuple[str, ...] | None = None
    holdout_fraction: float = 0.20
    max_train_rows: int = 20000
    max_validation_rows: int = 10000
    min_train_rows: int = 200
    min_validation_rows: int = 30
    max_iter: int = 80
    max_cpu_threads: int = 4
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    learning_rate: float = 0.05
    l2_regularization: float = 3.0
    residual_shrinkage: float | None = 0.5
    anomaly_quantile: float = 0.995
    scale_epsilon: float = 1e-8
    random_seed: int = 42
    evaluate_test: bool = True

    def validate(self) -> None:
        if not self.horizon_minutes or any(h <= 0 for h in self.horizon_minutes):
            raise ValueError("forecast horizons must be positive actual-time offsets")
        if not self.context_lags_minutes or any(x < 0 for x in self.context_lags_minutes):
            raise ValueError("context lags must be nonnegative")
        if self.seasonal_period_minutes <= 0 or (self.thinning_minutes is not None and self.thinning_minutes <= 0):
            raise ValueError("sampling/seasonal intervals must be positive")
        if not 0 < self.train_fraction < self.train_fraction + self.validation_fraction < 1:
            raise ValueError("chronological fractions must leave train, validation and test")
        if not 0 < self.train_scale_fraction < 1 or not 0 < self.validation_selection_fraction < 1:
            raise ValueError("calibration partitions must leave nonempty fit/selection periods")
        if not 0 <= self.holdout_fraction < 1 or not 0 < self.anomaly_quantile < 1:
            raise ValueError("invalid holdout fraction or anomaly quantile")
        if min(self.max_train_rows, self.max_validation_rows, self.min_train_rows,
               self.min_validation_rows, self.max_iter, self.min_samples_leaf) < 1:
            raise ValueError("sample/model budgets must be positive")
        if self.max_cpu_threads < 1:
            raise ValueError("max_cpu_threads must be positive")
        if self.max_train_rows < self.min_train_rows or self.max_validation_rows < self.min_validation_rows:
            raise ValueError("sample caps must accommodate minimum sample counts")
        if self.scale_epsilon <= 0:
            raise ValueError("normalisation epsilon must be positive")
        if self.residual_shrinkage is not None and not 0 < self.residual_shrinkage <= 1:
            raise ValueError("residual shrinkage must be in (0, 1], or None to disable that candidate")


def _utc(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise", format="mixed")
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamps must carry an explicit timezone; naive times are not assumed UTC")
    return parsed.dt.tz_convert("UTC")


def _boundary(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("split boundaries must carry an explicit timezone")
    return timestamp.tz_convert("UTC")


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_value(payload), indent=2, allow_nan=False), encoding="utf-8")


def _prepare_identity(frame: pd.DataFrame, cfg: RealObservationConfig):
    required = {"observation_id", *KEYS, *CHANNELS,
                *(f"{channel}__qc_accepted" for channel in CHANNELS)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing observation/quality fields: " + repr(sorted(missing)))
    if frame.empty:
        raise ValueError("no observations were supplied")
    work = frame.copy(deep=True)
    for column in ("observation_id", "station_id", "source"):
        if work[column].isna().any() or work[column].astype(str).str.len().eq(0).any():
            raise ValueError(f"{column} must be present; identifiers are not inferred")
        work[column] = work[column].astype(str)
    if work.observation_id.duplicated().any():
        raise ValueError("observation_id must identify an immutable original record uniquely")
    work["timestamp"] = _utc(work["timestamp"])
    if work.duplicated(list(KEYS)).any():
        raise ValueError("ambiguous duplicate station/source/timestamps; select a documented native report view")
    work = work.sort_values(list(KEYS), kind="stable").reset_index(drop=True)
    input_rows = len(work)
    if cfg.thinning_minutes is not None:
        # Select an existing report. No mean, interpolation or artificial grid.
        buckets = work["timestamp"].dt.floor(f"{cfg.thinning_minutes}min")
        keep = ~pd.DataFrame({"source": work.source, "station_id": work.station_id,
                              "bucket": buckets}).duplicated()
        work = work.loc[keep].reset_index(drop=True)
    dates = pd.DatetimeIndex(work.timestamp.unique()).sort_values()
    if len(dates) < 6:
        raise ValueError("not enough distinct observed times for chronological separation")
    if (cfg.train_end is None) != (cfg.validation_end is None):
        raise ValueError("set both train_end and validation_end, or neither")
    if cfg.train_end is None:
        train_end = dates[max(1, int(len(dates) * cfg.train_fraction))]
        validation_end = dates[min(len(dates) - 1, max(2, int(len(dates) * (cfg.train_fraction + cfg.validation_fraction))))]
    else:
        train_end, validation_end = _boundary(cfg.train_end), _boundary(cfg.validation_end)
    if train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")
    if cfg.test_end is not None:
        test_end = _boundary(cfg.test_end)
        if test_end <= validation_end:
            raise ValueError("test_end must follow validation_end")
        work = work.loc[work.timestamp < test_end].copy()
    else:
        test_end = None
    split = np.where(work.timestamp < train_end, "train",
                     np.where(work.timestamp < validation_end, "validation", "test"))
    work["evaluation_split"] = split
    stations = sorted(work.station_id.unique())
    if cfg.holdout_station_ids is not None:
        holdout = sorted(set(cfg.holdout_station_ids))
        if not set(holdout) <= set(stations):
            raise ValueError("holdout station IDs are absent from supplied observations")
    elif len(stations) > 1 and cfg.holdout_fraction > 0:
        count = min(len(stations) - 1, max(1, round(len(stations) * cfg.holdout_fraction)))
        holdout = sorted(np.random.default_rng(cfg.random_seed).choice(stations, count, replace=False).tolist())
    else:
        holdout = []
    if set(holdout) == set(stations):
        raise ValueError("station holdout cannot include every station")
    work["station_partition"] = np.where(work.station_id.isin(holdout), "holdout", "seen")
    split_counts = {name: int((work.evaluation_split == name).sum()) for name in ("train", "validation", "test")}
    if not split_counts["train"] or not split_counts["validation"]:
        raise ValueError("chronological boundaries must contain train and validation observations")
    # A selection-only experiment does not inspect test target values or QC.
    if not cfg.evaluate_test:
        work = work.loc[work.evaluation_split != "test"].copy()
    work = work.reset_index(drop=True)
    for channel in CHANNELS:
        work[channel] = pd.to_numeric(work[channel], errors="raise")
        for suffix in ("__qc_accepted", "__qc_rejected"):
            column = channel + suffix
            if column not in work:
                continue
            values = work[column].dropna()
            if not all(isinstance(x, (bool, np.bool_)) for x in values):
                raise ValueError(f"{column} must be nullable booleans with provider-defined semantics")
            work[column] = work[column].astype("boolean")
        rejected = channel + "__qc_rejected"
        if rejected in work and (work[channel + "__qc_accepted"] & work[rejected]).fillna(False).any():
            raise ValueError(f"contradictory provider acceptance/rejection for {channel}")
    return work, {"input_rows": input_rows, "retained_rows": len(work),
                  "sampling": "first existing report per clock interval" if cfg.thinning_minutes else "all supplied reports",
                  "thinning_minutes": cfg.thinning_minutes, "split_rows": split_counts,
                  "train_end_exclusive": train_end, "validation_end_exclusive": validation_end,
                  "test_end_exclusive": test_end, "holdout_station_ids": holdout,
                  "test_values_inspected": cfg.evaluate_test}


def build_causal_features(frame: pd.DataFrame, horizon_minutes: int,
                          context_lags_minutes=(0, 60, 180, 1440),
                          seasonal_period_minutes=1440) -> dict:
    """Exact timestamp joins; absent reports remain absent feature windows.

    Features for target time t end at forecast origin t-horizon. Provider QC is
    returned only as an eligibility mask and is never a model predictor.
    """
    if horizon_minutes <= 0 or any(x < 0 for x in context_lags_minutes):
        raise ValueError("horizon must be positive and context lags nonnegative")
    seasonal_target_lag = int(np.ceil(horizon_minutes / seasonal_period_minutes)) * seasonal_period_minutes
    offsets = sorted({0, *context_lags_minutes, seasonal_target_lag - horizon_minutes})
    lookup = frame.set_index(list(KEYS))
    if not lookup.index.is_unique:
        raise ValueError("causal features require unique source/station/timestamp records")
    values, lineage, accepted, names = [], {}, [], []
    origin = frame.timestamp - pd.Timedelta(minutes=horizon_minutes)
    for offset in offsets:
        times = origin - pd.Timedelta(minutes=offset)
        keys = pd.MultiIndex.from_arrays([frame.source, frame.station_id, times], names=KEYS)
        context = lookup.reindex(keys)
        block = context[list(CHANNELS)].to_numpy(dtype=float, na_value=np.nan)
        values.append(block)
        accepted.append(context[[c + "__qc_accepted" for c in CHANNELS]].fillna(False).to_numpy(bool).all(axis=1))
        lineage[f"origin_minus_{offset}m"] = context.observation_id.to_numpy()
        names.extend(f"{channel}__origin_minus_{offset}m" for channel in CHANNELS)
    X = np.column_stack(values)
    # Differences and cyclic time are derived predictors, separate from inputs.
    for i, offset in enumerate(offsets[1:], start=1):
        X = np.column_stack([X, values[0] - values[i]])
        names.extend(f"{channel}__change_over_{offset}m" for channel in CHANNELS)
    hour = origin.dt.hour.to_numpy() + origin.dt.minute.to_numpy() / 60.0
    day = origin.dt.dayofyear.to_numpy() - 1 + hour / 24.0
    for label, cycle in (("origin_hour", hour / 24.0), ("origin_dayofyear", day / 365.2425)):
        X = np.column_stack([X, np.sin(2 * np.pi * cycle), np.cos(2 * np.pi * cycle)])
        names.extend((label + "__sin", label + "__cos"))
    return {"X": X, "feature_names": names, "complete": np.isfinite(X).all(axis=1),
            "context_provider_accepted": np.column_stack(accepted).all(axis=1),
            "origin": origin, "lineage": lineage, "offsets_minutes": offsets,
            "persistence": values[0],
            "seasonal": values[offsets.index(seasonal_target_lag - horizon_minutes)]}


def _time_partition(frame: pd.DataFrame, mask: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.DatetimeIndex(frame.loc[mask, "timestamp"].unique()).sort_values()
    if len(dates) < 2:
        return mask.copy(), np.zeros(len(frame), bool)
    cut = dates[min(len(dates) - 1, max(1, int(len(dates) * fraction)))]
    return mask & (frame.timestamp < cut).to_numpy(), mask & (frame.timestamp >= cut).to_numpy()


def _balanced_cap(indices: np.ndarray, groups: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    rng = np.random.default_rng(seed)
    blocks = [rng.permutation(indices[groups[indices] == group]) for group in sorted(set(groups[indices]))]
    sampled, offset = [], 0
    while len(sampled) < cap:
        for block in blocks:
            if offset < len(block):
                sampled.append(int(block[offset]))
                if len(sampled) == cap:
                    break
        offset += 1
    return np.sort(np.asarray(sampled, dtype=int))


def _forecast_metrics(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict:
    valid = np.isfinite(y) & np.isfinite(prediction)
    if not valid.any():
        return {"status": "unavailable", "n": 0, "mae": None, "rmse": None, "macro_group_mae": None}
    error = prediction[valid] - y[valid]
    per_group = {}
    for group in sorted(set(groups[valid])):
        local = error[groups[valid] == group]
        per_group[group] = {"n": len(local), "mae": float(np.mean(np.abs(local))),
                            "rmse": float(np.sqrt(np.mean(local ** 2)))}
    return {"status": "available", "n": len(error), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "macro_group_mae": float(np.mean([v["mae"] for v in per_group.values()])),
            "per_group": per_group}


def _group_statistics(values, indices, groups, minimum_rows, statistic):
    """Fit local calibration only on the explicitly supplied historical rows."""
    mapping, counts = {}, {}
    for group in sorted(set(groups[indices])):
        local = indices[groups[indices] == group]
        local = local[np.isfinite(values[local])]
        counts[group] = len(local)
        if len(local) >= minimum_rows:
            mapping[group] = float(statistic(values[local]))
    return mapping, counts


def _group_values(groups, mapping, fallback):
    return np.asarray([mapping.get(group, fallback) for group in groups], dtype=float)


def _predict(candidate: str, bundle: dict, features: dict, channel_index: int, indices: np.ndarray) -> np.ndarray:
    if candidate == "persistence":
        return features["persistence"][indices, channel_index]
    if candidate == "seasonal_observed":
        return features["seasonal"][indices, channel_index]
    if candidate == "robust_change":
        return features["persistence"][indices, channel_index] + bundle["median_change"]
    with bundle["threadpool_controller"].limit(limits=bundle["max_cpu_threads"]):
        if candidate.startswith("hist_gradient_boosting_residual"):
            change = bundle["hist_gradient_boosting_residual"].predict(features["X"][indices])
            shrinkage = bundle["residual_shrinkage"] if candidate.endswith("_shrunk") else 1.0
            return features["persistence"][indices, channel_index] + shrinkage * change
        return bundle["hist_gradient_boosting"].predict(features["X"][indices])


def _provider_qc_agreement(frame, channel, flags, metadata):
    evidence = (metadata or {}).get("qc_agreement", {})
    if not (evidence.get("documented") is True and all(evidence.get(k) for k in
            ("positive_definition", "negative_definition", "evidence"))):
        return {"status": "unavailable", "reason": "documented provider-QC evaluation definitions were not supplied"}
    rejected = channel + "__qc_rejected"
    if rejected not in frame:
        return {"status": "unavailable", "reason": "no separately documented provider-rejection column"}
    positive = frame[rejected].fillna(False).to_numpy(bool)
    negative = frame[channel + "__qc_accepted"].fillna(False).to_numpy(bool)
    known = (positive | negative) & flags.notna().to_numpy()
    predicted = flags.fillna(False).to_numpy(bool)
    tp, fp = int((known & positive & predicted).sum()), int((known & negative & predicted).sum())
    fn, tn = int((known & positive & ~predicted).sum()), int((known & negative & ~predicted).sum())
    return {"status": "available" if known.any() else "unavailable", "task": "provider_QC_agreement_not_hardware_fault_truth",
            "definitions": evidence, "known_scored_rows": int(known.sum()),
            "unknown_or_unscored_rows": int((~known).sum()), "provider_rejected_flagged": tp,
            "provider_accepted_flagged": fp, "provider_rejected_unflagged": fn,
            "provider_accepted_unflagged": tn,
            "precision_against_provider_qc": tp / (tp + fp) if tp + fp else None,
            "recall_against_provider_qc": tp / (tp + fn) if tp + fn else None}


def run_real_observation_pipeline(observations: pd.DataFrame, out_dir: str | Path,
                                  config: RealObservationConfig | None = None,
                                  *, source_metadata: dict | None = None) -> dict:
    """Fit on accepted real observations and save honest forecasting evidence.

    ``evaluate_test=False`` performs model selection and calibration on the
    training/validation periods only. It does not access test target values or
    QC. Test target evaluation happens only after selecting models on validation.
    Missing fault labels are never created, encoded as normal, or scored.
    """
    cfg = config or RealObservationConfig()
    cfg.validate()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError("use a new experiment output directory to preserve previous evidence")
    frame, coverage = _prepare_identity(observations, cfg)
    groups = (frame.station_id + "|" + frame.source).to_numpy()
    seen = (frame.station_partition == "seen").to_numpy()
    train = (frame.evaluation_split == "train").to_numpy() & seen
    validation = (frame.evaluation_split == "validation").to_numpy() & seen
    train_fit, train_scale = _time_partition(frame, train, 1 - cfg.train_scale_fraction)
    validation_selection, validation_calibration = _time_partition(frame, validation, cfg.validation_selection_fraction)
    fingerprint_columns = ["observation_id", *KEYS, *CHANNELS,
                           *(c + "__qc_accepted" for c in CHANNELS),
                           *(c + "__qc_rejected" for c in CHANNELS if c + "__qc_rejected" in frame)]
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(
        frame[fingerprint_columns], index=False).to_numpy().tobytes()).hexdigest()
    report = {"schema_version": 1, "experiment": "real_observation_forecasting_and_residual_scoring",
              "data_fingerprint": {"sha256": fingerprint, "scheme": "pandas.hash_pandas_object over evaluated observation/QC columns, then SHA256",
                                   "columns": fingerprint_columns, "test_values_included": cfg.evaluate_test},
              "config": asdict(cfg), "coverage": coverage, "source_metadata": source_metadata or {},
              "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                          "scikit_learn": sklearn_version},
              "protocol": {"fault_labels": "not inferred; absent/unknown remain unknown",
                           "training_quality": "provider-accepted context and target values; not certified fault-free",
                           "feature_policy": "actual measured context values, their differences, and cyclic origin hour/day-of-year; QC excluded",
                           "missingness": "incomplete actual-time feature windows are unscored; no observation fills",
                           "model_selection": "lowest macro station|source MAE on first validation partition, seen stations only",
                           "baseline_comparison": "all candidates use the same complete accepted context/target rows",
                           "group_key": "station_id|source",
                           "residual_scale": "per-group median absolute forecast error on reserved chronological training tail; pooled fallback for unseen/insufficient groups",
                           "threshold": "per-group fixed quantile on later validation partition; accepted seen-station observations; pooled normalized-score fallback for unseen/insufficient groups",
                           "flag_rate": "model alert fraction; not false-positive rate or confirmed fault rate",
                           "test": "evaluated once after validation selection" if cfg.evaluate_test else "not evaluated; target values and QC not inspected"},
              "horizons": {}}
    scored = frame.copy(deep=True)
    models, trace = {}, {}
    candidates = ("persistence", "seasonal_observed", "robust_change", "hist_gradient_boosting",
                  "hist_gradient_boosting_residual")
    if cfg.residual_shrinkage is not None:
        candidates += ("hist_gradient_boosting_residual_shrunk",)
    threadpool_controller = ThreadpoolController()
    for horizon in cfg.horizon_minutes:
        features = build_causal_features(frame, horizon, cfg.context_lags_minutes, cfg.seasonal_period_minutes)
        hkey = f"{horizon}m"
        model_info, horizon_trace = {}, {}
        horizon_report = {"feature_names": features["feature_names"], "context_offsets_minutes": features["offsets_minutes"],
                          "complete_feature_rows": int(features["complete"].sum()),
                          "unavailable_feature_rows": int((~features["complete"]).sum()), "channels": {}}
        scored[hkey + "__forecast_origin"] = features["origin"]
        scored[hkey + "__context_provider_accepted"] = features["context_provider_accepted"]
        scored[hkey + "__features_available"] = features["complete"]
        for label, identifiers in features["lineage"].items():
            scored[f"{hkey}__context_id__{label}"] = identifiers
        channel_scores, aggregate_accepted = [], features["context_provider_accepted"].copy()
        for channel_index, channel in enumerate(CHANNELS):
            prefix = f"{hkey}__{channel}"
            y = frame[channel].to_numpy(float, na_value=np.nan)
            target_accepted = frame[channel + "__qc_accepted"].fillna(False).to_numpy(bool)
            eligible = features["complete"] & features["context_provider_accepted"] & target_accepted & np.isfinite(y)
            aggregate_accepted &= target_accepted & np.isfinite(y)
            raw_fit, raw_select = np.flatnonzero(train_fit & eligible), np.flatnonzero(validation_selection & eligible)
            fit = _balanced_cap(raw_fit, groups, cfg.max_train_rows, cfg.random_seed)
            select = _balanced_cap(raw_select, groups, cfg.max_validation_rows, cfg.random_seed)
            scale_rows = _balanced_cap(np.flatnonzero(train_scale & eligible), groups, cfg.max_validation_rows, cfg.random_seed)
            calibration = _balanced_cap(np.flatnonzero(validation_calibration & eligible), groups, cfg.max_validation_rows, cfg.random_seed)
            counts = {"eligible_training_fit_rows": len(raw_fit), "sampled_training_fit_rows": len(fit),
                      "eligible_validation_selection_rows": len(raw_select), "sampled_validation_selection_rows": len(select),
                      "sampled_training_scale_rows": len(scale_rows), "sampled_validation_calibration_rows": len(calibration),
                      "unavailable_targets": int((~np.isfinite(y)).sum())}
            prediction, score = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
            row_scales, row_thresholds = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
            flags = pd.Series(pd.NA, index=frame.index, dtype="boolean")
            channel_report = {"counts": counts}
            horizon_trace[channel] = {name: frame.observation_id.iloc[pos].tolist() for name, pos in
                                      (("fit", fit), ("training_scale", scale_rows), ("validation_selection", select), ("validation_calibration", calibration))}
            if len(fit) < cfg.min_train_rows or len(select) < cfg.min_validation_rows:
                channel_report.update(status="unavailable", reason="insufficient provider-accepted fit/selection targets")
            else:
                bundle = {"median_change": float(np.median(y[fit] - features["persistence"][fit, channel_index])),
                          "max_cpu_threads": cfg.max_cpu_threads, "threadpool_controller": threadpool_controller,
                          "residual_shrinkage": cfg.residual_shrinkage}
                parameters = dict(loss="absolute_error", learning_rate=cfg.learning_rate,
                    max_iter=cfg.max_iter, max_leaf_nodes=cfg.max_leaf_nodes, min_samples_leaf=cfg.min_samples_leaf,
                    l2_regularization=cfg.l2_regularization, early_stopping=False, random_state=cfg.random_seed)
                estimator = HistGradientBoostingRegressor(**parameters)
                residual_estimator = HistGradientBoostingRegressor(**parameters)
                # Both terms are original measurements; the difference is a
                # derived forecasting target, never a replacement observation.
                observed_change = y[fit] - features["persistence"][fit, channel_index]
                with threadpool_controller.limit(limits=cfg.max_cpu_threads):
                    estimator.fit(features["X"][fit], y[fit])
                    residual_estimator.fit(features["X"][fit], observed_change)
                bundle["hist_gradient_boosting"] = estimator
                bundle["hist_gradient_boosting_residual"] = residual_estimator
                validation_metrics = {name: _forecast_metrics(y[select], _predict(name, bundle, features, channel_index, select), groups[select])
                                      for name in candidates}
                selected = min(candidates, key=lambda name: validation_metrics[name]["macro_group_mae"])
                is_residual = selected.startswith("hist_gradient_boosting_residual")
                selected_estimator = residual_estimator if is_residual else estimator if selected == "hist_gradient_boosting" else None
                selected_bundle = {"selected_model": selected, "median_change": bundle["median_change"],
                                   "estimator": selected_estimator,
                                   "prediction_base": "observed forecast-origin reading" if is_residual else None,
                                   "predicted_change_multiplier": cfg.residual_shrinkage if selected.endswith("_shrunk") else 1.0,
                                   "feature_names": features["feature_names"], "residual_scale": None, "threshold": None,
                                   "per_group_residual_scales": {}, "per_group_thresholds": {}}
                available = np.flatnonzero(features["complete"])
                prediction[available] = _predict(selected, bundle, features, channel_index, available)
                channel_report.update(status="trained", selected_model=selected, validation_candidates=validation_metrics,
                                      residual_target_definition="observed target minus observed forecast-origin reading",
                                      selected_hyperparameters={**selected_estimator.get_params(),
                                          "prediction_base": "observed forecast-origin reading" if is_residual else None,
                                          "predicted_change_multiplier": selected_bundle["predicted_change_multiplier"]}
                                      if selected_estimator is not None else
                                      {"median_change": bundle["median_change"]} if selected == "robust_change" else {})
                if len(scale_rows) >= cfg.min_validation_rows:
                    absolute_residual = np.abs(y - prediction)
                    scale = max(float(np.median(absolute_residual[scale_rows])), cfg.scale_epsilon)
                    group_scales, group_scale_counts = _group_statistics(absolute_residual, scale_rows, groups,
                        cfg.min_validation_rows, lambda x: max(float(np.median(x)), cfg.scale_epsilon))
                    row_scales = _group_values(groups, group_scales, scale)
                    scored_rows = features["complete"] & np.isfinite(y)
                    score[scored_rows] = absolute_residual[scored_rows] / row_scales[scored_rows]
                    selected_bundle["residual_scale"] = scale
                    selected_bundle["per_group_residual_scales"] = group_scales
                    channel_report["residual_scale"] = scale
                    channel_report["per_group_residual_scales"] = group_scales
                    channel_report["per_group_scale_rows"] = group_scale_counts
                    if len(calibration) >= cfg.min_validation_rows:
                        threshold = float(np.quantile(score[calibration], cfg.anomaly_quantile))
                        group_thresholds, group_calibration_counts = _group_statistics(score, calibration, groups,
                            cfg.min_validation_rows, lambda x: np.quantile(x, cfg.anomaly_quantile))
                        row_thresholds = _group_values(groups, group_thresholds, threshold)
                        selected_bundle["threshold"] = threshold
                        selected_bundle["per_group_thresholds"] = group_thresholds
                        channel_report["threshold"] = threshold
                        channel_report["per_group_thresholds"] = group_thresholds
                        channel_report["per_group_calibration_rows"] = group_calibration_counts
                        channel_report["fallback"] = "pooled training residual scale / pooled validation normalized-score quantile for absent group mappings"
                        flags.loc[scored_rows] = score[scored_rows] > row_thresholds[scored_rows]
                    else:
                        channel_report["threshold_status"] = "unavailable_insufficient_validation_calibration"
                else:
                    channel_report["score_status"] = "unavailable_insufficient_training_scale_targets"
                model_info[channel] = selected_bundle
                channel_report["evaluation"] = {}
                for split_name in ("validation", "test"):
                    if split_name == "test" and not cfg.evaluate_test:
                        channel_report["evaluation"]["test"] = {"status": "not_evaluated"}
                        continue
                    channel_report["evaluation"][split_name] = {}
                    for partition in ("seen", "holdout"):
                        mask = (frame.evaluation_split == split_name).to_numpy() & (frame.station_partition == partition).to_numpy()
                        accepted_rows = np.flatnonzero(mask & eligible)
                        comparison = {name: _forecast_metrics(y[accepted_rows], _predict(name, bundle, features, channel_index, accepted_rows), groups[accepted_rows])
                                      if len(accepted_rows) else _forecast_metrics(np.array([]), np.array([]), np.array([]))
                                      for name in candidates}
                        raw_scored = mask & np.isfinite(score)
                        flagged = flags.loc[mask].dropna()
                        channel_report["evaluation"][split_name][partition] = {
                            "provider_accepted_forecasting": comparison,
                            "observed_rows": int(mask.sum()), "raw_scored_rows": int(raw_scored.sum()),
                            "unavailable_rows": int((mask & ~np.isfinite(score)).sum()),
                            "flagged_rows": int(flagged.sum()), "flag_rate": float(flagged.mean()) if len(flagged) else None,
                            "provider_qc_agreement": _provider_qc_agreement(frame.loc[mask].reset_index(drop=True), channel,
                                                                          flags.loc[mask].reset_index(drop=True), source_metadata)}
            scored[prefix + "__prediction"] = prediction
            scored[prefix + "__absolute_error"] = np.where(np.isfinite(y) & np.isfinite(prediction), np.abs(y - prediction), np.nan)
            scored[prefix + "__anomaly_score"] = score
            scored[prefix + "__residual_scale"] = row_scales
            scored[prefix + "__threshold"] = row_thresholds
            scored[prefix + "__flag"] = flags
            channel_scores.append(score)
            horizon_report["channels"][channel] = channel_report
        scores = np.column_stack(channel_scores)
        complete_scores = np.isfinite(scores).all(axis=1)
        aggregate_score = np.full(len(frame), np.nan)
        aggregate_score[complete_scores] = np.max(scores[complete_scores], axis=1)
        aggregate_calibration = _balanced_cap(np.flatnonzero(validation_calibration & aggregate_accepted & complete_scores),
                                              groups, cfg.max_validation_rows, cfg.random_seed)
        aggregate_flag = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        threshold, aggregate_group_thresholds, aggregate_group_counts = None, {}, {}
        aggregate_row_thresholds = np.full(len(frame), np.nan)
        if len(aggregate_calibration) >= cfg.min_validation_rows:
            threshold = float(np.quantile(aggregate_score[aggregate_calibration], cfg.anomaly_quantile))
            aggregate_group_thresholds, aggregate_group_counts = _group_statistics(aggregate_score, aggregate_calibration, groups,
                cfg.min_validation_rows, lambda x: np.quantile(x, cfg.anomaly_quantile))
            aggregate_row_thresholds = _group_values(groups, aggregate_group_thresholds, threshold)
            aggregate_flag.loc[complete_scores] = aggregate_score[complete_scores] > aggregate_row_thresholds[complete_scores]
        scored[hkey + "__anomaly_score"] = aggregate_score
        scored[hkey + "__threshold"] = aggregate_row_thresholds
        scored[hkey + "__flag"] = aggregate_flag
        horizon_report["aggregate"] = {"rule": "maximum of all three available channel residual scores",
                                       "threshold": threshold, "calibration_rows": len(aggregate_calibration),
                                       "per_group_thresholds": aggregate_group_thresholds,
                                       "per_group_calibration_rows": aggregate_group_counts,
                                       "fallback": "pooled validation normalized-score quantile for absent group thresholds",
                                       "scored_rows": int(complete_scores.sum()), "unavailable_rows": int((~complete_scores).sum()),
                                       "flag_rate": float(aggregate_flag.dropna().mean()) if aggregate_flag.notna().any() else None}
        aggregate_evaluation = {}
        for split_name in ("validation", "test"):
            if split_name == "test" and not cfg.evaluate_test:
                aggregate_evaluation[split_name] = {"status": "not_evaluated"}
                continue
            aggregate_evaluation[split_name] = {}
            for partition in ("seen", "holdout"):
                mask = ((frame.evaluation_split == split_name)
                        & (frame.station_partition == partition)).to_numpy()
                observed_flags = aggregate_flag.loc[mask].dropna()
                aggregate_evaluation[split_name][partition] = {
                    "observed_rows": int(mask.sum()), "scored_rows": int((mask & complete_scores).sum()),
                    "unavailable_score_rows": int((mask & ~complete_scores).sum()),
                    "thresholded_rows": len(observed_flags), "flagged_rows": int(observed_flags.sum()),
                    "flag_rate": float(observed_flags.mean()) if len(observed_flags) else None}
        horizon_report["aggregate"]["evaluation"] = aggregate_evaluation
        models[hkey] = {"channels": model_info, "aggregate_threshold": threshold,
                        "per_group_aggregate_thresholds": aggregate_group_thresholds,
                        "context_offsets_minutes": features["offsets_minutes"]}
        horizon_trace["aggregate_calibration"] = frame.observation_id.iloc[aggregate_calibration].tolist()
        trace[hkey] = horizon_trace
        report["horizons"][hkey] = horizon_report
    report["trained_channel_models"] = sum(len(h["channels"]) for h in models.values())
    report["expected_channel_models"] = len(CHANNELS) * len(cfg.horizon_minutes)
    report["status"] = ("completed" if report["trained_channel_models"] == report["expected_channel_models"]
                        else "completed_partial" if report["trained_channel_models"] else "unavailable_no_models_fit")
    report["artifacts"] = {"scored_observations": str(output / "scored_observations.parquet"),
                           "models": str(output / "forecast_models.joblib"), "training_trace": str(output / "training_trace.json")}
    scored.to_parquet(output / "scored_observations.parquet", index=False)
    joblib.dump({"schema_version": 1, "config": asdict(cfg), "models": models,
                 "data_fingerprint": report["data_fingerprint"]}, output / "forecast_models.joblib")
    _write_json(output / "training_trace.json", trace)
    _write_json(output / "config.json", {**asdict(cfg), "resolved_protocol": coverage})
    _write_json(output / "metrics.json", report)
    return _json_value(report)
