"""SkyGuard AI — score a new AWS data feed with trained artifacts.

    python scripts/run_inference.py --input new_station.csv --artifacts artifacts_full

Input CSV: `timestamp` + the three measured channels (+ anything else is
ignored). Output: `scored.csv` (per-observation score & flag) and `alerts.csv`
(events with root-cause classification, confidence, explanation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from awsad.explain import Explainer                                     # noqa: E402
from awsad.models.ensemble import WeightedEnsemble                      # noqa: E402
from awsad.models.fault_classifier import classify_event                # noqa: E402
from awsad.models.isolation_forest import IsolationForestDetector       # noqa: E402
from awsad.preprocessing.features import (DETECTION_CHANNELS, MEASURED_CHANNELS,  # noqa: E402
                                          add_physics_derived,
                                          add_window_features,
                                          apply_climatology,
                                          feature_columns,
                                          fit_climatology,
                                          impute_hourly,
                                          robust_scale_apply,
                                          robust_scale_fit)
from awsad.preprocessing.qc_rules import compute_qc_flags, qc_score     # noqa: E402


def resolve_station(df: pd.DataFrame, group: str | None, meta_groups: list[str]) -> str | None:
    if group:
        return group
    if "station_id" in df.columns:
        sid = str(df["station_id"].iloc[0])
        for g in meta_groups:
            if g.startswith(sid):
                return g
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--artifacts", default="artifacts_full_classical")
    ap.add_argument("--group", default=None, help="station|source key for per-station calibration")
    ap.add_argument("--out", default="out_inference")
    ap.add_argument("--ref-hours", type=int, default=24 * 30,
                    help="clean leading window used as reference for unseen stations")
    ap.add_argument("--no-deep", action="store_true", help="skip deep heads even if present")
    args = ap.parse_args()

    art = Path(args.artifacts)
    ens = WeightedEnsemble.load(str(art / "ensemble.json"))
    meta = json.loads((art / "feature_meta.json").read_text())
    rcd_all = joblib.load(art / "robust_channel.joblib")
    use_if = (art / "isolation_forest.joblib").exists()
    ifdet = IsolationForestDetector.load(str(art / "isolation_forest.joblib")) if use_if else None

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    for c in MEASURED_CHANNELS:
        if c not in df.columns:
            raise SystemExit(f"input must contain column '{c}'")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_physics_derived(df)

    meta_groups = list(meta["groups"].keys()) if meta and "groups" in meta else []
    group = resolve_station(df, args.group, meta_groups)
    if group in meta_groups:
        print(f"matched station group: {group}")
        gmeta = meta["groups"][group]
    else:
        if group:
            print(f"group '{group}' has no trained station metadata; treating it as unseen")
        group = None
        print("unseen station — calibrating reference on leading clean window")
        ref = df.iloc[: min(len(df), args.ref_hours)]
        clim = fit_climatology(ref, DETECTION_CHANNELS)
        scale = robust_scale_fit(
            add_window_features(apply_climatology(ref, DETECTION_CHANNELS, clim),
                                DETECTION_CHANNELS, use_aux=False),
            feature_columns(add_window_features(apply_climatology(ref, DETECTION_CHANNELS, clim),
                                                DETECTION_CHANNELS, use_aux=False),
                            DETECTION_CHANNELS))
        gmeta = {"climatology": clim, "robust_scale": scale}

    cols = [c for c in DETECTION_CHANNELS if c in df.columns]
    df_full = df.copy()
    df_full["station_id"] = df_full.get("station_id", "custom")
    # hourly regularization (buffering assumption for streams)
    df = impute_hourly(df_full.sort_values("timestamp").reset_index(drop=True),
                       [c for c in MEASURED_CHANNELS if c in df_full.columns])
    df = add_physics_derived(df)

    # ---- component scores
    det_cols = [c for c in df.columns if c in set(DETECTION_CHANNELS) or c == "timestamp"]
    qcs = qc_score(compute_qc_flags(df[det_cols])).to_numpy()
    clim = gmeta["climatology"]
    featurized = add_window_features(apply_climatology(df, cols, clim), cols, use_aux=False)
    resid = pd.DataFrame({c: featurized[f"{c}__resid"] for c in cols
                          if f"{c}__resid" in featurized.columns})

    rcd = rcd_all.get(group) if group in rcd_all else None
    if rcd is None:
        ref_len = min(len(df), args.ref_hours)
        ref_r = resid.iloc[:ref_len]
        ref_raw = df[cols].iloc[:ref_len]
        from awsad.models.statistical import RobustChannelDetector
        rcd = RobustChannelDetector().fit(ref_raw, ref_r)
    zss = rcd.score(df[cols], resid)

    scores = {"qc": qcs, "zscore": zss}
    fcols_all = meta.get("feature_cols") or []
    for c in fcols_all:
        if c not in featurized.columns:
            featurized[c] = np.nan
    scaled_df = robust_scale_apply(featurized, gmeta["robust_scale"])
    scaled_df = scaled_df.reindex(columns=fcols_all, fill_value=0.0)
    if ifdet is not None:
        scores["iforest"] = ifdet.score(scaled_df)

    # ---- deep heads (optional, load if the artifacts include them)
    if not args.no_deep:
        block = scaled_df[fcols_all].to_numpy(dtype=np.float32)
        block = np.nan_to_num(block, nan=0.0)
        for name in ("lstm_ae_24", "tx_ae", "forecaster", "lstm_ae_168"):
            mp = art / f"{name}.pt"
            if not mp.exists():
                continue
            try:
                if name.startswith("lstm_ae"):
                    from awsad.models.lstm_autoencoder import LSTMAEDetector
                    det = LSTMAEDetector.load(str(mp), device="cpu")
                elif name == "tx_ae":
                    from awsad.models.transformer_ae import TransformerAEDetector
                    det = TransformerAEDetector.load(str(mp), device="cpu")
                else:
                    from awsad.models.lstm_forecaster import ForecasterDetector
                    det = ForecasterDetector.load(str(mp), device="cpu")
                scores[name] = det.score_series(block)
                print(f"  scored with deep head: {name}")
            except Exception as e:
                print(f"  ! {name} failed to load/score: {e}")

    # match the batch pipeline's score shaping: EWMA sustain + pointwise max
    for k in list(scores):
        s = pd.Series(scores[k]).ewm(halflife=3.0, adjust=False).mean().to_numpy()
        scores[k] = np.maximum(s, scores[k])

    final = ens.score(scores, ens.per_group_norms.get(group) if group else None)
    thr = ens.threshold_for(group)
    # deterministic physics overrides (frozen sensor / impossible physics always alert)
    from awsad.preprocessing.qc_rules import hard_flag
    hard = hard_flag(compute_qc_flags(df[det_cols])).to_numpy()
    df["anomaly_score"] = final
    df["is_anomaly_soft"] = (final >= thr).astype(int)
    df["qc_hard_flag"] = hard.astype(int)
    df["is_anomaly"] = np.maximum(df["is_anomaly_soft"], hard).astype(int)
    df["confidence"] = 1.0 / (1.0 + np.exp(-2.0 * (final / max(thr, 1e-9) - 1.0)))
    df.loc[hard, "confidence"] = np.maximum(df.loc[hard, "confidence"], 0.98)

    # ---- events + explanations
    labels = df["is_anomaly"].to_numpy()
    bnd = np.flatnonzero(np.diff(np.concatenate(([0], labels, [0]))))
    events = []
    explainer = Explainer()
    qc_all = compute_qc_flags(df[det_cols])
    for s, e in zip(bnd[0::2], bnd[1::2]):
        ctx = df.iloc[max(0, s - 168):s]
        info = classify_event(ctx, df.iloc[s:e], qc_window=qc_all.iloc[s:e])
        attr = explainer.channel_attribution(df.iloc[s:e], ctx,
                                             resid.iloc[s:e], rcd.params)
        reason = explainer.reason_string(info, attr)
        events.append({"start": df.timestamp.iloc[s], "end": df.timestamp.iloc[e - 1],
                       "peak_score": float(final[s:e].max()),
                       **info, "explanation": reason})
    evdf = pd.DataFrame(events)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scored.csv", index=False)
    evdf.to_csv(out / "alerts.csv", index=False)
    print(f"\n{int(labels.sum())} anomalous hours in {len(events)} events  "
          f"(threshold {thr:.3f})")
    print(f"Wrote {out/'scored.csv'} and {out/'alerts.csv'}")
    if len(evdf):
        print("\nSample explanation:")
        print(" ", evdf.iloc[0]["explanation"])


if __name__ == "__main__":
    main()
