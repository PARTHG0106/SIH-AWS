"""SkyGuard AI — shared training + evaluation pipeline (SIH26073).

Single source of truth used by scripts/run_local_train.py AND the Kaggle
training notebook (offline).

Stages per (station, source) group:
    QC physics rules            ('qc')
    Robust channel detector     ('zscore')   — residual levels + dynamics
    Isolation Forest            ('iforest')  — pruned windowed features
    LSTM autoencoder x2 windows ('lstm_ae_24', 'lstm_ae_168')
    Transformer autoencoder     ('tx_ae')
    LSTM forecaster             ('forecaster')
    Spatial buddy-check         ('spatial')
Final: per-station normalisation, Dirichlet weight calibration on the labeled
validation split, per-station supervised thresholds (POT fallback for unseen
stations), pooled + per-group evaluation, per-fault breakdown, holdout-station
generalization, leave-one-out ablation, score cache for dashboards/notebooks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data.stations import INDIAN_STATIONS
from .data.training_contract import require_eligible_training_data
from .evaluation.calibration import temporal_partitions, balanced_group_class_weights
from .evaluation.metrics import (aucs, best_weight_objective_over_quantiles,
                                 choose_threshold, mean_detection_latency,
                                 point_adjust, pointwise, range_wise)
from .evaluation.thresholds import pot_threshold
from .models.ensemble import WeightedEnsemble
from .models.fault_classifier import classify_event
from .models.isolation_forest import IsolationForestDetector
from .models.spatial import SpatialConsistency
from .models.statistical import RobustChannelDetector
from .preprocessing.features import (DETECTION_CHANNELS, add_window_features,
                                     apply_climatology, feature_columns,
                                     fit_climatology, robust_scale_apply,
                                     robust_scale_fit)
from .preprocessing.qc_rules import compute_qc_flags, hard_flag, qc_score

# --------------------------------------------------------------------------- io


def load_processed(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    p = Path(processed_dir)
    require_eligible_training_data(p)

    def rd(name):
        pq, cg = p / f"{name}.parquet", p / f"{name}.csv.gz"
        if pq.exists():
            return pd.read_parquet(pq)
        return pd.read_csv(cg, parse_dates=["timestamp"])

    out = {"train": rd("train_clean"), "val": rd("val_labeled"),
           "test": rd("test_labeled")}
    ev = p / "injection_events.parquet"
    out["events"] = pd.read_parquet(ev) if ev.exists() else pd.DataFrame()
    splits = p / "splits.json"
    out["splits"] = json.loads(splits.read_text()) if splits.exists() else {}
    stats = p / "station_stats.json"
    out["station_stats"] = json.loads(stats.read_text()) if stats.exists() else {}
    catalog = p / "station_catalog.json"
    if catalog.exists():
        out["station_catalog"] = json.loads(catalog.read_text())
    else:
        embedded = out["splits"].get("station_catalog", []) \
            if isinstance(out["splits"], dict) else []
        out["station_catalog"] = embedded
    source_meta = p / "source_metadata.json"
    out["source_metadata"] = (json.loads(source_meta.read_text())
                               if source_meta.exists() else {})
    out["processed_dir"] = str(p.resolve())
    return out


def _get_group(df: pd.DataFrame, key: str) -> pd.DataFrame:
    sid, src = key.split("|")
    m = (df["station_id"].astype(str) == sid) & (df["source"].astype(str) == src)
    return df[m]


def split_groups(dfs: dict[str, pd.DataFrame], stations=None, sources=None) -> list[str]:
    keys = sorted(set(zip(dfs["train"]["station_id"].astype(str),
                          dfs["train"]["source"].astype(str))))
    keys = [f"{a}|{b}" for a, b in keys]
    if stations:
        keys = [k for k in keys if k.split("|")[0] in stations]
    if sources:
        keys = [k for k in keys if k.split("|")[1] in sources]
    return keys


def _resid_frame(feat_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = {}
    for c in cols:
        if f"{c}__resid" in feat_df.columns:
            out[c] = feat_df[f"{c}__resid"]
    res = pd.DataFrame(out, index=feat_df.index)
    if "timestamp" in feat_df.columns:
        res["timestamp"] = feat_df["timestamp"]
    return res


def select_if_features(train_f: pd.DataFrame, fcols: list[str],
                       max_missing: float = 0.5) -> list[str]:
    keep = []
    for c in fcols:
        s = train_f[c]
        if s.isna().mean() > max_missing:
            continue
        if np.nanstd(s.to_numpy(dtype=float)) < 1e-3 and not c.endswith("__was_missing"):
            continue
        keep.append(c)
    return keep


def select_if_features_blocks(blocks: list[pd.DataFrame], fcols: list[str],
                              max_missing: float = 0.5) -> list[str]:
    """Select IF columns using all fit groups without concatenating them.

    The old implementation inspected only a small probe of stations to keep
    RAM bounded.  With heterogeneous sources that probe can be unrepresentative
    (for example, a channel absent in the first eight stations but present in
    most of the network).  Accumulating finite counts and Welford moments keeps
    the memory O(number of features) while making the decision network-wide.
    """
    if not blocks:
        return []
    total = {c: 0 for c in fcols}
    finite_n = {c: 0 for c in fcols}
    mean = {c: 0.0 for c in fcols}
    m2 = {c: 0.0 for c in fcols}
    for block in blocks:
        n = len(block)
        for c in fcols:
            if c not in block:
                total[c] += n
                continue
            x = block[c].to_numpy(dtype=np.float64, copy=False, na_value=np.nan)
            ok = np.isfinite(x)
            vals = x[ok]
            total[c] += n
            if not len(vals):
                continue
            finite_n[c] += len(vals)
            # Merge this block's moments into the running population moments.
            bmean = float(vals.mean())
            bm2 = float(np.sum((vals - bmean) ** 2))
            old_n = finite_n[c] - len(vals)
            if old_n <= 0:
                mean[c], m2[c] = bmean, bm2
            else:
                delta = bmean - mean[c]
                new_n = old_n + len(vals)
                m2[c] += bm2 + delta * delta * old_n * len(vals) / new_n
                mean[c] += delta * len(vals) / new_n
    keep = []
    for c in fcols:
        if total[c] <= 0 or finite_n[c] / total[c] < (1.0 - max_missing):
            continue
        var = m2[c] / max(finite_n[c] - 1, 1)
        if np.sqrt(max(var, 0.0)) < 1e-3 and not c.endswith("__was_missing"):
            continue
        keep.append(c)
    # Isolation Forest cannot fit an empty matrix.  Keep one finite feature as
    # a conservative fallback for very sparse smoke bundles.
    if not keep:
        for c in fcols:
            if finite_n[c] > 0:
                keep = [c]
                break
    return keep


def _ewma_max(a: np.ndarray, halflife: float = 3.0) -> np.ndarray:
    """max(EWMA, raw) with full NaN/±Inf sanitisation — a single polluted point
    must never poison the trailing EWMA (or the ensemble)."""
    if np.isfinite(a).any():
        finite_max = float(np.nanmax(a[np.isfinite(a)])) if np.isfinite(a).any() else 1.0
    else:
        finite_max = 1.0
    a = np.nan_to_num(a, nan=0.0, posinf=finite_max, neginf=0.0)
    s = pd.Series(a).ewm(halflife=halflife, adjust=False).mean().to_numpy()
    return np.maximum(s, a)


# ---------------------------------------------------------------------- pipeline
def run_pipeline(dfs: dict[str, pd.DataFrame], groups: list[str],
                 out_dir: str | Path, *,
                 deep_cfg: dict | None = None,
                 if_cfg: dict | None = None,
                 train_deep: bool = True,
                 log=print) -> dict:
    if not dfs.get("processed_dir"):
        raise ValueError("training requires a verified processed_dir; in-memory data lack source evidence")
    require_eligible_training_data(dfs["processed_dir"])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    report: dict = {"groups": list(groups), "artifacts": {},
                    "data_provenance": dfs.get("source_metadata", {})}
    split_meta = dfs.get("splits") or {}
    core_cols = [c for c in DETECTION_CHANNELS]
    splits = ("train", "val", "test")
    # Build group indices once. Repeated full-corpus string conversions/scans
    # previously dominated preprocessing time for hundreds of stations.
    row_index = {
        s: {f"{sid}|{source}": idx for (sid, source), idx in
            dfs[s].groupby(["station_id", "source"], sort=False, observed=True).indices.items()}
        for s in splits
    }

    # Groups must have enough history in every split; off-network stations
    # (e.g. opened after 2023) simply can't be calibrated.
    requested_groups = list(groups)
    pre = len(requested_groups)
    groups = [g for g in requested_groups
              if all(len(row_index[s].get(g, [])) >= 2000 for s in splits)]
    if pre != len(groups):
        log(f"filtered groups without full-split coverage: {pre} -> {len(groups)}")
    if not groups:
        raise ValueError("no station|source group has at least 2000 rows in train, val and test")

    # Establish the holdout protocol only after coverage filtering.  This
    # avoids advertising holdouts that cannot actually be evaluated and keeps
    # the fit set valid for small smoke selections.
    holdout_stations = {str(s) for s in split_meta.get("holdout_stations", [])}
    holdout_groups = [g for g in groups if g.split("|", 1)[0] in holdout_stations]
    fit_groups = [g for g in groups if g not in holdout_groups]
    if holdout_groups and not fit_groups:
        raise ValueError("all selected groups are marked as holdouts; include at least one fit group")
    if not fit_groups:
        fit_groups = list(groups)
    fit_set = set(fit_groups)
    report["groups"] = list(groups)
    report["protocol"] = {
        "holdout_stations": sorted(holdout_stations),
        "fit_groups": fit_groups,
        "holdout_groups": holdout_groups,
        "station_local_adaptation": (
            "climatology, robust scaling and component normalization use only each "
            "group's clean training history; holdout groups are excluded from global "
            "model fitting and all label-based calibration"
        ),
    }
    log(f"Protocol: {len(fit_groups)} fit groups, {len(holdout_groups)} holdout groups")

    ds: dict[str, list[pd.DataFrame]] = {s: [] for s in splits}
    R: dict[str, dict[str, pd.DataFrame]] = {s: {} for s in splits}
    metas: dict[str, dict] = {}
    rcd_models: dict[str, RobustChannelDetector] = {}
    QC: dict[str, dict[str, np.ndarray]] = {s: {} for s in splits}
    QCH: dict[str, dict[str, np.ndarray]] = {s: {} for s in splits}
    fcols: list[str] | None = None
    if_fcols: list[str] | None = None

    log("== featurizing ==")
    for g in groups:
        gg = {s: dfs[s].iloc[row_index[s][g]].reset_index(drop=True) for s in splits}
        for s in splits:
            if "label" not in gg[s]:
                gg[s]["label"] = np.nan
            ts = gg[s]["timestamp"]
            if ts.isna().any() or not ts.is_monotonic_increasing:
                raise ValueError(f"{s}/{g}: timestamps must be chronological and nonmissing")
        if not (gg["train"]["timestamp"].max() < gg["val"]["timestamp"].min()
                and gg["val"]["timestamp"].max() < gg["test"]["timestamp"].min()):
            raise ValueError(f"{g}: train/validation/test timestamps overlap")
        clim = fit_climatology(gg["train"], core_cols)
        frames = {s: add_window_features(apply_climatology(gg[s], core_cols, clim),
                                         core_cols, use_aux=False)
                  for s in splits}
        if fcols is None:
            fcols = feature_columns(frames["train"], core_cols)
        rcd = RobustChannelDetector()
        rcd.fit(gg["train"], _resid_frame(frames["train"], core_cols))
        rcd_models[g] = rcd
        metas[g] = {"climatology": clim, "feature_cols": fcols,
                    "robust_scale": robust_scale_fit(frames["train"], fcols)}
        for s in splits:
            f = frames[s].copy(); f["group"] = g
            for c in fcols:
                if c not in f.columns:
                    f[c] = np.nan
            # memory discipline: float32 everywhere, slim column footprint
            f64 = f.select_dtypes("float64").columns
            if len(f64):
                f[f64] = f[f64].astype("float32")
            keep = ["timestamp", "station_id", "source", "group", "label",
                    "temperature_c", "pressure_hpa", "relative_humidity_pct",
                    "dewpoint_c", "td_spread", "vpd_hpa"]
            slim = [c for c in keep if c in f.columns] + list(fcols)
            ds[s].append(f[slim].copy())
            R[s][g] = _resid_frame(frames[s], core_cols)
            raw_core = gg[s][[c for c in gg[s].columns
                              if c == "timestamp" or c in set(core_cols)]]
            _fl = compute_qc_flags(raw_core)
            QC[s][g] = qc_score(_fl).to_numpy()
            QCH[s][g] = hard_flag(_fl).to_numpy()
    fit_blocks = [ds["train"][groups.index(g)] for g in fit_groups]
    if_fcols = select_if_features_blocks(fit_blocks, fcols)
    del fit_blocks
    for g in groups:
        metas[g]["if_feature_cols"] = if_fcols

    # Pooled robust-param fallback: a station whose own training slice is too
    # sparse (RobustChannelDetector skips channels <50% finite) would otherwise
    # get an EMPTY detector -> zero robust-z signal in the batch ensemble AND a
    # dead streaming detector.  Fill any missing channel with the network-median
    # robust stats so every group has a working per-channel detector.
    _seen_ch: set[str] = set()
    for g in fit_groups:
        _seen_ch |= set(rcd_models[g].params)
    _pool_params: dict[str, dict] = {}
    for c in _seen_ch:
        vals = [rcd_models[g].params[c] for g in fit_groups if c in rcd_models[g].params]
        if vals:
            _pool_params[c] = {k: float(np.median([v[k] for v in vals]))
                               for k in vals[0]}
    _filled_groups = 0
    for g in groups:
        p = rcd_models[g].params
        missing = [c for c in _pool_params if c not in p]
        if missing:
            _filled_groups += 1
            for c in missing:
                p[c] = dict(_pool_params[c])
    if _filled_groups:
        log(f"   robust-channel pooled fallback: filled missing params for "
            f"{_filled_groups}/{len(groups)} groups")
    joblib.dump(rcd_models, out_dir / "robust_channel.joblib")

    # in-place per-group robust scaling (avoids a second full-size copy)
    log("== scaling features per group ==")
    for s in splits:
        for i, g in enumerate(groups):
            sub = ds[s][i]
            sc = robust_scale_apply(sub, metas[g]["robust_scale"])
            sub.loc[:, fcols] = sc[fcols].to_numpy()
            ds[s][i] = sub

    y_g = {s: {g: dsSplit["label"].to_numpy(dtype=float, na_value=np.nan)
               for g, dsSplit in zip(groups, ds[s])} for s in ("val", "test")}

    # ------------------------------------------------------------ classical
    log("== robust channel detector + isolation forest ==")
    ZS = {s: {g: rcd_models[g].score(_get_group(dfs[s], g).reset_index(drop=True),
                                     R[s][g]) for g in groups} for s in splits}

    if_cfg = {"n_estimators": 300, "max_samples": 65536,
              "contamination": 0.03} | (if_cfg or {})
    # bounded random subsample across fit groups for IF training
    rng_fs = np.random.default_rng(1)
    fit_parts = []
    per_cap = max(2000, int(1_500_000 / max(len(fit_groups), 1)))
    for g in fit_groups:
        sub = ds["train"][groups.index(g)][if_fcols]
        if len(sub) > per_cap:
            ix = rng_fs.choice(len(sub), per_cap, replace=False)
            fit_parts.append(sub.iloc[ix])
        else:
            fit_parts.append(sub)
    tr_fit = pd.concat(fit_parts, ignore_index=True)
    IF = IsolationForestDetector(**{k: v for k, v in if_cfg.items()
                                    if k in ("n_estimators", "max_samples", "contamination")})
    IF.fit(tr_fit)
    del tr_fit, fit_parts
    IF.save(str(out_dir / "isolation_forest.joblib"))
    IFS = {s: {g: IF.score(ds[s][i][if_fcols]) for i, g in enumerate(groups)}
           for s in splits}

    # ------------------------------------------------------------ spatial
    log("== spatial consistency ==")
    catalog_blob = dfs.get("station_catalog") or {}
    catalog_rows = (catalog_blob.get("stations", [])
                    if isinstance(catalog_blob, dict) else catalog_blob)
    statmeta = {}
    for row in catalog_rows:
        try:
            sid = str(row.get("station_id") or f"{row['usaf']}{row['wban']}")
            lat, lon = float(row.get("lat", 0.0)), float(row.get("lon", 0.0))
            elev = float(row.get("elevation_m", row.get("elev", 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
        if lat == 0.0 and lon == 0.0:
            continue
        statmeta[sid] = {"lat": lat, "lon": lon, "elev": elev}
    # Old bundles do not contain station_catalog.json; retain the curated
    # fallback so those artifacts remain loadable.
    if not statmeta:
        statmeta = {s.station_id: {"lat": s.lat, "lon": s.lon,
                                  "elev": s.elevation_m}
                    for s in INDIAN_STATIONS}
    sp = SpatialConsistency(statmeta)
    sp.fit({g: R["train"][g] for g in fit_groups})
    with open(out_dir / "spatial.json", "w") as f:
        json.dump({"train_spread": {f"{g}|{c}": v for (g, c), v in sp.train_spread.items()},
                   "channel_spread": sp.channel_spread}, f)
    SP = {s: sp.score_all(R[s]) for s in splits}
    SP = {s: {g: SP[s].get(g, np.zeros(len(R[s][g]))) for g in groups} for s in splits}

    # ------------------------------------------------------------ deep models
    DEEP: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    if train_deep:
        DEEP.update(_train_deep(ds, groups, fit_groups, metas, fcols, out_dir,
                                deep_cfg or {}, log))

    # ------------------------------------------------------------ assemble S
    log("== ensemble calibration + thresholds ==")
    comp_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    comp_arrays["qc"] = QC
    comp_arrays["zscore"] = ZS
    comp_arrays["iforest"] = IFS
    comp_arrays["spatial"] = SP
    for name, blocks in DEEP.items():
        comp_arrays[name] = blocks

    S: dict[str, dict[str, dict[str, np.ndarray]]] = {s: {g: {} for g in groups}
                                                      for s in splits}
    for cname, per_split in comp_arrays.items():
        for s in splits:
            for g in groups:
                S[s][g][cname] = _ewma_max(per_split[s][g])
    comps = list(comp_arrays)

    norms = {g: {k: (float(np.nanmedian(S["train"][g][k])),
                     float(np.nanquantile(S["train"][g][k], 0.99))) for k in comps}
             for g in groups}

    def ens_g(split: str, g: str, weights: dict[str, float]) -> np.ndarray:
        tot, wsum = None, 0.0
        for k, w in weights.items():
            if w <= 0:
                continue
            m, h = norms[g][k]
            sn = (S[split][g][k] - m) / max(h - m, 1e-9)
            tot = sn * w if tot is None else tot + sn * w
            wsum += w
        if tot is None:
            # No positive-weight component (e.g. LOO ablation removing the only
            # weighted detector) -> a flat no-signal score, not a crash.
            length = len(next(iter(S[split][g].values())))
            return np.zeros(length, dtype=float)
        out = tot / max(wsum, 1e-9)
        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)

    # weight calibration (Dirichlet search, supervised on validation split).
    # Weights are GLOBAL, so a fixed random subset of fit groups carries the
    # same statistical signal while keeping the search O(minutes) at any scale.
    rng = np.random.default_rng(0)
    CAL_CAP = 64
    if len(fit_groups) > CAL_CAP:
        cal_groups = sorted(rng.choice(fit_groups, CAL_CAP, replace=False).tolist())
        log(f"   calibrating weights on {len(cal_groups)} of {len(fit_groups)} fit groups")
    else:
        cal_groups = list(fit_groups)
    cal_groups = [g for g in cal_groups
                  if np.any(y_g["val"][g] == 1) and np.any(y_g["val"][g] == 0)]
    val_norm = {g: {k: (S["val"][g][k] - norms[g][k][0])
                    / max(norms[g][k][1] - norms[g][k][0], 1e-9)
                    for k in comps} for g in cal_groups}
    y_val_cal = {g: y_g["val"][g] for g in cal_groups}
    # Cover interpretable sparse baselines before random mixtures. This avoids
    # a dense Dirichlet search missing a strong single detector or pair.
    weight_candidates = [np.ones(len(comps)) / len(comps)]
    weight_candidates.extend(np.eye(len(comps)))
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            w = np.zeros(len(comps))
            w[i] = w[j] = 0.5
            weight_candidates.append(w)
    weight_candidates.extend(
        rng.dirichlet(np.full(len(comps), 0.35), size=256)
    )
    weight_candidates.extend(
        rng.dirichlet(np.ones(len(comps)), size=256)
    )

    best_w = dict(zip(comps, np.ones(len(comps)) / len(comps)))
    best_objective = -1.0
    for w in weight_candidates if cal_groups else []:
        objective = float(np.mean([
            best_weight_objective_over_quantiles(
                sum(val_norm[g][k] * w_i for k, w_i in zip(comps, w)) / w.sum(),
                y_val_cal[g])[0]
            for g in cal_groups]))
        if objective > best_objective:
            best_objective, best_w = objective, dict(zip(comps, w))
    log(f"   weights: {{{', '.join(f'{k}={v:.3f}' for k, v in best_w.items())}}} "
        f"(val strict-weighted objective={best_objective:.3f}; "
        f"candidates={len(weight_candidates)})")
    report["weight_selection"] = {
        "method": "validation_strict_f1" if cal_groups else "unvalidated_equal_weight_baseline",
        "calibration_groups": cal_groups,
    }

    # Label-tuned thresholds are fit only on non-holdout validation groups.
    # Every group also gets a clean-train POT threshold for deployment without
    # labels; the pooled POT threshold handles a completely unseen group.
    thr_g = {}
    y_val_all = np.concatenate([y_g["val"][g] for g in fit_groups])
    ens_val_all = np.concatenate([ens_g("val", g, best_w) for g in fit_groups])
    ens_train_all = np.concatenate([ens_g("train", g, best_w) for g in fit_groups])
    thr_pot = pot_threshold(ens_train_all, 0.98, 1e-3)
    has_supervision = np.any(y_val_all == 1) and np.any(y_val_all == 0)
    thr_pool = choose_threshold(ens_val_all, y_val_all)[0] if has_supervision else thr_pot
    if has_supervision:
        for g in fit_groups:
            y = y_g["val"][g]
            local_support = np.count_nonzero(y == 1) >= 20 and np.count_nonzero(y == 0) >= 100
            thr_g[g] = choose_threshold(ens_g("val", g, best_w), y)[0] if local_support else thr_pool
    thr_pot_g = {g: pot_threshold(ens_g("train", g, best_w), 0.98, 1e-3)
                 for g in groups}
    # Streaming fast-path uses only (qc, zscore) — spatial/deep/stacker need
    # neighbours/windows/batch. The batch ensemble legitimately zeroes zscore
    # (redundant with iforest there), but for the streaming SUBSET the robust-z
    # level signal is the workhorse. Calibrate a dedicated (qc, zscore) blend on
    # validation so streaming actually uses it, instead of slicing best_w.
    _stream_cands = [{"qc": 1.0 - wz, "zscore": wz}
                     for wz in np.linspace(0.0, 1.0, 11)]
    _best_sw, _best_sobj = {"qc": 0.5, "zscore": 0.5}, -1.0
    for sw in _stream_cands if cal_groups else []:
        if not all(k in comps for k in sw):
            continue
        obj = float(np.mean([
            best_weight_objective_over_quantiles(ens_g("val", g, sw),
                                                 y_g["val"][g])[0]
            for g in cal_groups]))
        if obj > _best_sobj:
            _best_sobj, _best_sw = obj, sw
    stream_weights = _best_sw
    log(f"   streaming weights (qc/zscore blend): {stream_weights} "
        f"(val objective={_best_sobj:.3f})")

    def stream_g(split: str, g: str) -> np.ndarray:
        return ens_g(split, g, stream_weights)

    stream_thr_g = {g: choose_threshold(stream_g("val", g), y_g["val"][g])[0]
                    for g in fit_groups if np.count_nonzero(y_g["val"][g] == 1) >= 20
                    and np.count_nonzero(y_g["val"][g] == 0) >= 100}
    stream_pot_g = {g: pot_threshold(stream_g("train", g), 0.98, 1e-3)
                    for g in groups}
    stream_train_all = np.concatenate([stream_g("train", g) for g in fit_groups])
    stream_pot = pot_threshold(stream_train_all, 0.98, 1e-3)
    log(f"   pooled val-optimal thr={thr_pool:.3f} (POT fallback={thr_pot:.3f})")

    ens = WeightedEnsemble(best_w)
    ens.norms_ = {k: (float(np.nanmedian(np.concatenate([S["train"][g][k] for g in fit_groups]))),
                      float(np.nanquantile(np.concatenate([S["train"][g][k] for g in fit_groups]), 0.99)))
                  for k in comps}
    ens.per_group_norms = norms
    ens.per_group_thresholds = thr_g
    ens.per_group_fallback_thresholds = thr_pot_g
    ens.pooled_threshold = thr_pool
    ens.fallback_threshold = thr_pot
    ens.save(str(out_dir / "ensemble.json"))
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump({"ensemble": thr_pool, "ensemble_pot_fallback": thr_pot,
                   "per_group": thr_g, "per_group_pot_fallback": thr_pot_g},
                  f, indent=2)
    with open(out_dir / "stream_thresholds.json", "w") as f:
        json.dump({"components": stream_weights,
                   "pot_fallback": stream_pot,
                   "per_group": stream_thr_g,
                   "per_group_pot_fallback": stream_pot_g}, f, indent=2)
    with open(out_dir / "feature_meta.json", "w") as f:
        json.dump({"groups": {g: {"climatology": metas[g]["climatology"],
                                  "robust_scale": metas[g]["robust_scale"]}
                              for g in groups},
                   "feature_cols": fcols, "if_feature_cols": if_fcols}, f)
    if catalog_blob:
        with open(out_dir / "station_catalog.json", "w", encoding="utf-8") as f:
            json.dump(catalog_blob, f, indent=2)

    # ------------------------------------------------------------ evaluation
    log("== evaluation (test) ==")
    te = pd.concat([f[["timestamp", "station_id", "source", "group", "label",
                       "temperature_c", "pressure_hpa", "relative_humidity_pct",
                       "dewpoint_c", "td_spread", "vpd_hpa"]]
                    if "dewpoint_c" in f.columns else
                    f[["timestamp", "station_id", "source", "group", "label",
                       "temperature_c", "pressure_hpa", "relative_humidity_pct",
                       "td_spread", "vpd_hpa"]]
                    for f in ds["test"]], ignore_index=True)
    y_test = te["label"].to_numpy(dtype=float, na_value=np.nan)
    ens_test = np.concatenate([ens_g("test", g, best_w) for g in groups])
    pred_soft = np.concatenate([(ens_g("test", g, best_w) >= ens.threshold_for(g)).astype(int)
                                for g in groups])
    hard = np.concatenate([QCH["test"][g] for g in groups]).astype(int)
    pred = np.maximum(pred_soft, hard)         # deterministic rule override
    seen_mask = te["group"].isin(fit_set).to_numpy()
    holdout_mask = ~seen_mask
    report["test"] = {
        "ensemble_pooled_thr": _binary_report(
            y_test[seen_mask], ens_test[seen_mask],
            (ens_test[seen_mask] >= thr_pool).astype(int),
            "single pooled supervised threshold on non-holdout groups",
            te.loc[seen_mask, "group"].to_numpy()),
        "ensemble_per_group_thr": _binary_report(
            y_test, ens_test, pred_soft,
            "seen groups use supervised validation thresholds; holdouts use train-only POT",
            te["group"].to_numpy()),
        "ensemble_final": _binary_report(
            y_test, ens_test, pred,
            "production detector: mixed threshold policy plus deterministic physics overrides",
            te["group"].to_numpy()),
        "thresholds": {"pooled_supervised": thr_pool, "pot_fallback": thr_pot,
                       "supervised_groups": len(thr_g),
                       "pot_only_groups": len(groups) - len(thr_g),
                       "stream_pot_fallback": stream_pot},
        "weights": best_w}
    report["test"]["ensemble_pooled_thr"]["scope"] = "non-holdout groups only"
    _log_report(log, "ensemble pooled", report["test"]["ensemble_pooled_thr"])
    eg = report["test"]["ensemble_per_group_thr"]
    log(f"   [per-group thr] pointF1={eg['pointwise']['f1']:.3f} "
        f"paF1={eg['point_adjust']['f1']:.3f} rangeF1={eg['range_wise']['f1']:.3f}")
    ef = report["test"]["ensemble_final"]
    log(f"   [FINAL w/overrides] pointF1={ef['pointwise']['f1']:.3f} "
        f"paF1={ef['point_adjust']['f1']:.3f} rangeF1={ef['range_wise']['f1']:.3f}")

    for k in comps:
        pooled = np.concatenate([(S["test"][g][k] - norms[g][k][0])
                                 / max(norms[g][k][1] - norms[g][k][0], 1e-9)
                                 for g in groups])
        report["test"][f"single::{k}"] = _binary_report(
            y_test, pooled, (pooled >= 1.0).astype(int),
            "fixed normalized component threshold", te["group"].to_numpy())
        _log_report(log, k, report["test"][f"single::{k}"])

    # Leave-one-out ablation on non-holdout test groups at the fixed pooled
    # threshold. Strict pointwise F1 is the primary selection/reporting metric.
    report["test"]["ablation_loo"] = {}
    for k in comps:
        w = {c: v for c, v in best_w.items() if c != k}
        s = np.concatenate([ens_g("test", g, w) for g in fit_groups])
        y_seen = np.concatenate([y_g["test"][g] for g in fit_groups])
        seen_group_ids = np.concatenate([
            np.repeat(g, len(y_g["test"][g])) for g in fit_groups])
        r = _binary_report(y_seen, s, (s >= thr_pool).astype(int),
                           group_ids=seen_group_ids)["pointwise"]["f1"]
        report["test"]["ablation_loo"][k] = round(float(r), 4)
    log(f"   LOO ablation (pointF1 @pooled thr): {report['test']['ablation_loo']}")

    events = dfs.get("events", pd.DataFrame())
    if len(events):
        report["test"]["per_fault"] = per_fault_breakdown(events, te, pred)
        log("   per-fault recall: " +
            ", ".join(f"{k}={v['recall']}" for k, v in
                      sorted(report["test"]["per_fault"].items())))

    if seen_mask.any():
        report["test"]["seen"] = _binary_report(
            y_test[seen_mask], ens_test[seen_mask], pred[seen_mask],
            "groups used for global model fitting and supervised calibration",
            te.loc[seen_mask, "group"].to_numpy())
    if holdout_mask.any():
        report["test"]["holdout"] = _binary_report(
            y_test[holdout_mask], ens_test[holdout_mask], pred[holdout_mask],
            "excluded from global model fitting and label-based calibration; "
            "station-local clean-history adaptation plus POT only",
            te.loc[holdout_mask, "group"].to_numpy())
        _log_report(log, "holdout(ensemble)", report["test"]["holdout"])

    # ------------------------------------------------- supervised meta-learner
    # The validation period has three disjoint chronological purposes:
    # classifier fitting, iteration selection, then threshold calibration.
    # Holdout stations participate in none of them. Event purging and elapsed
    # time gaps prevent nearby windows sharing an event across these roles.
    try:
        import copy
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import average_precision_score
        from .inference import stack_meta_features

        _gidx = {g: i for i, g in enumerate(groups)}
        meta_fcols = list(fcols)

        def _Xn(split, g, rows=None):
            # [per-group-normalized component scores | robust-scaled fcols
            #  (residual dynamics incl. was_missing/roll_std) | QC hard flag]
            rows = slice(None) if rows is None else rows
            ns = np.column_stack([
                (S[split][g][k][rows] - norms[g][k][0])
                / max(norms[g][k][1] - norms[g][k][0], 1e-9) for k in comps])
            fm = ds[split][_gidx[g]].iloc[rows][meta_fcols].to_numpy(dtype=np.float32)
            return stack_meta_features(ns, fm, QCH[split][g][rows])

        partitions = {g: temporal_partitions(ds["val"][_gidx[g]]["timestamp"],
                                             y_g["val"][g]) for g in fit_groups}
        meta_rng = np.random.default_rng(42)

        def stage_rows(stage, cap):
            result = {}
            per_group = max(1, cap // len(fit_groups))
            for g in fit_groups:
                rows = partitions[g][stage]
                if len(rows) > per_group:
                    rows = np.sort(meta_rng.choice(rows, per_group, replace=False))
                if len(rows):
                    result[g] = rows
            return result

        fit_rows, tune_rows = stage_rows("fit", 500_000), stage_rows("tune", 250_000)
        calibration_rows = {g: partitions[g]["calibrate"] for g in fit_groups
                            if len(partitions[g]["calibrate"])}
        if not fit_rows or not tune_rows or not calibration_rows:
            raise ValueError("insufficient known labels after chronological purging")
        Xtr = np.vstack([_Xn("val", g, rows) for g, rows in fit_rows.items()])
        ytr = np.concatenate([y_g["val"][g][rows] for g, rows in fit_rows.items()])
        group_tr = np.concatenate([np.repeat(g, len(rows)) for g, rows in fit_rows.items()])
        if len(np.unique(ytr)) != 2:
            raise ValueError("stacker fit partition needs both reviewed label classes")
        tune = {g: (_Xn("val", g, rows), y_g["val"][g][rows])
                for g, rows in tune_rows.items()
                if len(np.unique(y_g["val"][g][rows])) == 2}
        if not tune:
            raise ValueError("no tuning group has both reviewed label classes")
        clf = HistGradientBoostingClassifier(
            max_iter=50, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=5.0,
            early_stopping=False, warm_start=True, random_state=42)
        sample_weight = balanced_group_class_weights(ytr, group_tr)
        best_clf, best_ap, history = None, -1.0, []
        for iterations in (50, 100, 200, 400):
            clf.set_params(max_iter=iterations)
            clf.fit(Xtr, ytr, sample_weight=sample_weight)
            ap = float(np.mean([average_precision_score(y, clf.predict_proba(x)[:, 1])
                                for x, y in tune.values()]))
            history.append({"iterations": iterations, "macro_pr_auc": ap})
            if ap > best_ap + 1e-4:
                best_clf, best_ap = copy.deepcopy(clf), ap
            else:
                break
        clf = best_clf
        del Xtr, ytr, group_tr, tune, sample_weight

        def _proba(split, g):
            return clf.predict_proba(_Xn(split, g))[:, 1]

        cal_scores = {g: clf.predict_proba(_Xn("val", g, rows))[:, 1]
                      for g, rows in calibration_rows.items()}
        cal_labels = {g: y_g["val"][g][rows] for g, rows in calibration_rows.items()}
        ycal = np.concatenate(list(cal_labels.values()))
        if len(np.unique(ycal)) != 2:
            raise ValueError("threshold calibration needs both reviewed label classes")
        thr_sup, _ = choose_threshold(np.concatenate(list(cal_scores.values())), ycal)
        # Small/one-class station slices use the independently calibrated
        # pooled threshold; they cannot support a reliable local optimum.
        thr_sup_g = {g: choose_threshold(cal_scores[g], y)[0]
                     for g, y in cal_labels.items()
                     if np.count_nonzero(y == 1) >= 20 and np.count_nonzero(y == 0) >= 100}
        report["stacker_calibration"] = {
            "status": "trained", "purge_hours": 168,
            "partition_fractions": [0.6, 0.2, 0.2],
            "selected_iterations": int(clf.n_iter_), "tuning_history": history,
            "rows": {g: {stage: len(rows) for stage, rows in parts.items()}
                     for g, parts in partitions.items()},
            "sampled_fit_rows": sum(map(len, fit_rows.values())),
            "sampled_tune_rows": sum(map(len, tune_rows.values())),
            "probability_semantics": "class-balanced classifier score; not calibrated fault probability",
        }
        sup_test = {g: _proba("test", g) for g in groups}
        sup_score = np.concatenate([sup_test[g] for g in groups])
        sup_pred = np.concatenate([
            (sup_test[g] >= thr_sup_g.get(g, thr_sup)).astype(int) for g in groups])
        # Match SupervisedScorer.predict exactly; deterministic overrides have
        # their own ensemble_final report and are not silently added here.
        report["test"]["ensemble_supervised"] = _binary_report(
            y_test[seen_mask], sup_score[seen_mask], sup_pred[seen_mask],
            "supervised stacker, threshold-only serving policy (seen groups)",
            te.loc[seen_mask, "group"].to_numpy())
        _log_report(log, "SUPERVISED(seen)", report["test"]["ensemble_supervised"])
        if holdout_mask.any():
            report["test"]["ensemble_supervised_holdout"] = _binary_report(
                y_test[holdout_mask], sup_score[holdout_mask], sup_pred[holdout_mask],
                "supervised stacker on holdout stations (pooled threshold)",
                te.loc[holdout_mask, "group"].to_numpy())
            _log_report(log, "SUPERVISED(holdout)",
                        report["test"]["ensemble_supervised_holdout"])
        if len(events):
            report["test"]["per_fault_supervised"] = per_fault_breakdown(
                events, te, sup_pred)
            log("   supervised per-fault recall: " + ", ".join(
                f"{k}={v['recall']}" for k, v in
                sorted(report["test"]["per_fault_supervised"].items())))
        import joblib as _joblib
        _joblib.dump({"model": clf, "components": comps, "fcols": meta_fcols,
                      "pooled_threshold": float(thr_sup),
                      "per_group_thresholds": thr_sup_g,
                      "calibration_protocol": report["stacker_calibration"]},
                     out_dir / "supervised_meta.joblib")
    except ValueError as _e:
        report["stacker_calibration"] = {"status": "skipped", "reason": str(_e)}
        log(f"   [supervised meta-learner skipped: {_e!r}]")

    # ------------------------------------------------------------ persist
    cache = {}
    for s in splits:
        for g in groups:
            for k in comps:
                cache[f"{s}|{g}|{k}"] = S[s][g][k]
        for g in groups:
            i = groups.index(g)
            cache[f"{s}|{g}|label"] = ds[s][i]["label"].to_numpy() if "label" in ds[s][i] else np.full(len(ds[s][i]), np.nan)
            cache[f"{s}|{g}|ts"] = ds[s][i]["timestamp"].astype("int64").to_numpy()
    np.savez_compressed(out_dir / "scores_cache.npz", **cache)
    report["runtime_s"] = round(time.time() - t_start, 1)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    alerts = attribute_events(te, pred, ens_test)
    alerts.to_csv(out_dir / "demo_alerts.csv", index=False)
    log(f"== DONE in {report['runtime_s']}s -> {out_dir} ==")
    return report


# ------------------------------------------------------------------ deep models
def _train_deep(ds, groups, fit_groups, metas, fcols, out_dir, deep_cfg, log):
    from .models.lstm_autoencoder import LSTMAEDetector, TrainConfig
    from .models.transformer_ae import TransformerAEDetector, TransformerTrainConfig
    from .models.lstm_forecaster import ForecasterDetector

    scaled = {s: [ds[s][i][fcols].to_numpy(np.float32, copy=True)
                  for i in range(len(groups))] for s in ("train", "val", "test")}
    fit_idx = [groups.index(g) for g in fit_groups]
    train_blocks = [scaled["train"][i] for i in fit_idx]
    # Validation labels are synthetic anomalies, so using the full validation
    # frame for early stopping would reward reconstruction of corrupted inputs.
    # Deep heads instead validate on a clean chronological tail of TRAIN.
    deep_train_blocks, deep_val_blocks = [], []
    for block in train_blocks:
        cut = max(1, int(len(block) * 0.9))
        if len(block) - cut < 256:
            deep_train_blocks.append(block)
        else:
            deep_train_blocks.append(block[:cut])
            deep_val_blocks.append(block[cut:])

    heads = {
        "lstm_ae_24": ("lstm", 24), "lstm_ae_168": ("lstm", 168),
        "tx_ae": ("tx", 48), "forecaster": ("fore", 24),
    }
    overrides = deep_cfg or {}
    results = {}
    for name, (kind, w) in heads.items():
        cfgd = overrides.get(name, {})
        if not cfgd.get("enabled", True):
            continue
        log(f"-- deep head {name} (window={w}) --")
        # Shared knobs are intentionally exposed through the notebook config:
        # on a large GPU we can increase capacity/batch size, while balanced
        # group sampling and a robust loss remain the safe defaults.
        common = dict(
            epochs=cfgd.get("epochs", 30),
            batch_size=cfgd.get("batch_size", 256),
            lr=cfgd.get("lr", 1e-3),
            patience=cfgd.get("patience", 5),
            windows_per_epoch=cfgd.get("windows_per_epoch", 65536),
            val_windows=cfgd.get("val_windows", 8192),
            balance_groups=cfgd.get("balance_groups", True),
            loss=cfgd.get("loss", "huber"),
            huber_beta=cfgd.get("huber_beta", 1.0),
        )
        if kind == "lstm":
            det = LSTMAEDetector(window=w, hidden=cfgd.get("hidden", 64),
                                 layers=cfgd.get("layers", 2),
                                 dropout=cfgd.get("dropout", 0.1))
            cfg = TrainConfig(window=w, **common)
        elif kind == "tx":
            det = TransformerAEDetector(
                window=w,
                d_model=cfgd.get("d_model", 128),
                nhead=cfgd.get("nhead", 8),
                layers=cfgd.get("layers", 3),
                ff=cfgd.get("ff", 256),
                bottleneck=cfgd.get("bottleneck", 64),
                dropout=cfgd.get("dropout", 0.1),
            )
            cfg = TransformerTrainConfig(window=w, **common)
        else:
            det = ForecasterDetector(
                window=w,
                hidden=cfgd.get("hidden", 96),
                layers=cfgd.get("layers", 2),
                dropout=cfgd.get("dropout", 0.1),
            )
            cfg = TrainConfig(window=w, **common)
        det.fit(deep_train_blocks, fcols,
                val_blocks=deep_val_blocks or None, cfg=cfg, log=log)
        det.save(str(out_dir / f"{name}.pt"))
        results[name] = {
            s: {g: det.score_series(b) for g, b in zip(groups, scaled[s])}
            for s in ("train", "val", "test")}
    return results


# ------------------------------------------------------------------ helpers
def _scale_all_groups(df: pd.DataFrame, metas: dict) -> pd.DataFrame:
    out = df.copy()
    gcol = "group" if "group" in df.columns else None
    if gcol:
        for gname, idx in out.groupby(gcol, sort=False).groups.items():
            if gname not in metas:
                raise KeyError(f"missing scaling metadata for group {gname}")
            fcols = metas[gname]["feature_cols"]
            scaled = robust_scale_apply(out.loc[idx], metas[gname]["robust_scale"])
            # Assign by the original index so concatenated frames never reorder.
            out.loc[idx, fcols] = scaled[fcols].to_numpy()
        return out
    if len(metas) != 1:
        raise ValueError("cannot determine group for scaling")
    gname = next(iter(metas))
    fcols = metas[gname]["feature_cols"]
    scaled = robust_scale_apply(out, metas[gname]["robust_scale"])
    out.loc[:, fcols] = scaled[fcols].to_numpy()
    return out


def _binary_report(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray,
                   note: str | None = None,
                   group_ids: np.ndarray | None = None) -> dict:
    """Report on known labels without inventing negatives at group boundaries."""
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.ndim != 1 or scores.shape != y_true.shape or y_pred.shape != y_true.shape:
        raise ValueError("labels, scores and predictions must be aligned vectors")
    known = np.isin(y_true, [0, 1])
    out = {
        "n_rows": len(y_true), "n_known_labels": int(known.sum()),
        "n_unknown_labels": int((~known).sum()),
        "n_unavailable_scores": int((~np.isfinite(scores)).sum()),
        "anomaly_rate_pred": float(y_pred.mean()) if len(y_pred) else float("nan"),
        "anomaly_rate_true": float(y_true[known].mean()) if known.any() else float("nan"),
        "pointwise": pointwise(y_true, y_pred),
    }
    out.update(aucs(y_true, scores))
    # Only event diagnostics get separators. Separator rows must never enter
    # prevalence, AUC, counts or pointwise scores.
    event_y = np.where(known, y_true, np.nan)
    event_pred = np.where(known, y_pred, 0)
    if group_ids is not None and len(y_true) > 1:
        group_ids = np.asarray(group_ids)
        if group_ids.shape != y_true.shape:
            raise ValueError("group IDs must align with labels")
        cuts = np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1
        if len(cuts):
            event_y = np.insert(event_y, cuts, np.nan)
            event_pred = np.insert(event_pred, cuts, 0)
        # Traverse contiguous runs instead of rescanning the full test corpus
        # once per station for every detector/ablation report.
        totals = {}
        bounds = np.r_[0, cuts, len(y_true)]
        for start, end in zip(bounds[:-1], bounds[1:]):
            group = str(group_ids[start])
            local_known = known[start:end]
            yt, yp = y_true[start:end][local_known], y_pred[start:end][local_known]
            if not len(yt):
                continue
            stat = np.array([len(yt), np.sum((yt == 1) & (yp == 1)),
                             np.sum((yt == 0) & (yp == 1)),
                             np.sum((yt == 1) & (yp == 0))])
            totals[group] = totals.get(group, np.zeros(4, dtype=int)) + stat
        group_reports, counts = {}, []
        for group, (n, tp, fp, fn) in totals.items():
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            group_reports[group] = {"n_known_labels": int(n), "precision": float(p),
                                     "recall": float(r), "f1": float(2 * p * r / (p + r)) if p + r else 0.0}
            counts.append([tp, fp, fn])
        out["per_group_pointwise"] = group_reports
        if group_reports:
            out["macro_group_f1"] = float(np.mean([v["f1"] for v in group_reports.values()]))
        if len(counts) >= 2:
            count_array = np.asarray(counts)
            rng = np.random.default_rng(42)
            draw = rng.integers(0, len(counts), size=(500, len(counts)))
            sums = count_array[draw].sum(axis=1)
            denom = 2 * sums[:, 0] + sums[:, 1] + sums[:, 2]
            boot = np.divide(2 * sums[:, 0], denom, out=np.zeros(500), where=denom > 0)
            out["point_f1_group_bootstrap_95ci"] = np.quantile(boot, [.025, .975]).tolist()
    out.update({
        "point_adjust": point_adjust(event_y, event_pred),
        "range_wise": range_wise(event_y, event_pred),
        "mean_latency_steps": mean_detection_latency(event_y, event_pred),
    })
    if note:
        out["note"] = note
    return out


def _log_report(log, name, r):
    pw, pa, rw = r["pointwise"], r["point_adjust"], r["range_wise"]
    log(f"   [{name:22s}] pointF1={pw['f1']:.3f}  pointAdjF1={pa['f1']:.3f}  "
        f"rangeF1={rw['f1']:.3f}  PR-AUC={r.get('pr_auc', float('nan')):.3f}  "
        f"P={pw['precision']:.3f} R={pw['recall']:.3f}")


def per_fault_breakdown(events, te, pred):
    # Index each station/source once. The old implementation converted and
    # scanned all test station IDs for every event (minutes at corpus scale).
    lookup = {}
    pred = np.asarray(pred)
    for (station, source), pos in te.groupby(["station_id", "source"], sort=False,
                                            observed=True).indices.items():
        stamps = pd.DatetimeIndex(pd.to_datetime(te.iloc[pos]["timestamp"], utc=True))
        if not stamps.is_monotonic_increasing:
            raise ValueError("event evaluation requires chronological group rows")
        lookup[(str(station), str(source))] = (
            stamps.asi8, stamps.unit, np.r_[0, np.cumsum(pred[pos] == 1)])
    out = {}
    for _, ev in events.iterrows():
        item = lookup.get((str(ev["station_id"]), str(ev["source"])))
        if item is None:
            continue
        stamps, unit, hits = item
        # DatetimeIndex storage may be us on pandas 3; use the same resolution
        # for boundaries rather than assuming .asi8 always contains ns.
        start = pd.Timestamp(ev["start"]).as_unit(unit).asm8.view("i8")
        end = pd.Timestamp(ev["end"]).as_unit(unit).asm8.view("i8")
        left = np.searchsorted(stamps, start, side="left")
        right = np.searchsorted(stamps, end, side="right")
        if left == right:
            continue
        per = out.setdefault(ev["fault"], {"n_events": 0, "detected": 0})
        per["n_events"] += 1
        per["detected"] += int(hits[right] > hits[left])
    for f, d in out.items():
        d["recall"] = round(d["detected"] / d["n_events"], 3) if d["n_events"] else None
    return out


def attribute_events(te, pred, scores, max_events=300):
    rows = []
    te = te.reset_index(drop=True)
    for group, idx in te.groupby("group", sort=False).groups.items():
        if len(rows) >= max_events:
            break
        pos = np.asarray(list(idx), dtype=int)
        local_pred = np.asarray(pred)[pos]
        local_scores = np.asarray(scores)[pos]
        local = te.iloc[pos].reset_index(drop=True)
        bnd = np.flatnonzero(np.diff(np.concatenate(([0], local_pred, [0]))))
        for s, e in zip(bnd[0::2], bnd[1::2]):
            if len(rows) >= max_events:
                break
            ctx = local.iloc[max(0, s - 168):s]
            info = classify_event(ctx, local.iloc[s:e])
            rows.append({"group": group,
                         "start": local.loc[s, "timestamp"],
                         "end": local.loc[e - 1, "timestamp"],
                         "peak_score": float(local_scores[s:e].max()), **info})
    return pd.DataFrame(rows)
