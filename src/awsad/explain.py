"""Explainability for SkyGuard AI alerts (SIH26073: 'Explainable AI-based
reasoning', 'root-cause classification').

For each detected event we produce:
  * channel attribution (which parameter drives the anomaly, by residual share)
  * component contribution (which model supported the flag)
  * physics consistency evidence (QC rules fired, Magnus closure violation)
  * a human-readable reason string for operators
  * optional IF permutation importance (global, computed once on train)

SHAP on the Isolation Forest is used when the `shap` package is importable
(present in Kaggle's image); otherwise we fall back to permutation importance —
the presented API is identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNEL_LABEL = {
    "temperature_c": "Temperature", "pressure_hpa": "Atmospheric pressure",
    "relative_humidity_pct": "Relative humidity", "dewpoint_c": "Dew point",
    "td_spread": "T-Td spread", "vpd_hpa": "Vapour-pressure deficit",
    "es_hpa": "Saturation vapour pressure", "press_tendency_3h": "Pressure tendency",
}


class Explainer:
    def __init__(self, norms: dict[str, float] | None = None):
        self.norms = norms or {}

    # ------------------------------------------------------------- per event
    def channel_attribution(self, raw_window: pd.DataFrame,
                            ref_frame: pd.DataFrame,
                            resid_window: pd.DataFrame,
                            rcd_params: dict[str, dict]) -> dict:
        """Rank channels by robust-z of residuals within the event window."""
        ranking = []
        for c, p in rcd_params.items():
            if c not in resid_window.columns:
                continue
            v = resid_window[c].to_numpy(dtype=float)
            if not np.isfinite(v).any():
                continue
            z = float(np.nanmedian(np.abs(v - p["med"]) / p["sigma"]))
            ranking.append({"channel": c,
                            "label": CHANNEL_LABEL.get(c, c),
                            "mean_abs_z": z,
                            "share": 0.0})
        total = sum(r["mean_abs_z"] for r in ranking) or 1.0
        for r in ranking:
            r["share"] = r["mean_abs_z"] / total
        ranking.sort(key=lambda r: -r["mean_abs_z"])

        # which raw channels moved most vs. seasonal reference?
        deltas = {}
        for c in ("temperature_c", "pressure_hpa", "relative_humidity_pct"):
            if c in raw_window.columns and c in ref_frame.columns:
                deltas[c] = float(np.nanmedian(raw_window[c])
                                  - np.nanmedian(ref_frame[c]))
        return {"channels": ranking, "raw_shift": deltas}

    def component_support(self, comp_scores: dict[str, np.ndarray],
                          weights: dict[str, float]) -> dict[str, float]:
        sup = {}
        for k, s in comp_scores.items():
            r = float(np.nanmean(s)) if len(s) else 0.0
            sup[k] = {"mean_score": r, "weight": weights.get(k, 0.0),
                      "weighted": r * weights.get(k, 0.0)}
        return sup

    def reason_string(self, fault: dict, attribution: dict,
                      qc_hits: dict[str, int] | None = None) -> str:
        parts = []
        ft = fault.get("fault_type", "unknown")
        ch = fault.get("channel", "")
        label = CHANNEL_LABEL.get(ch, ch)
        conf = fault.get("confidence", 0.5)
        sentences = {
            "spike": f"{label} shows an isolated impulse inconsistent with the station's own climate.",
            "drift": f"{label} exhibits a slow monotone drift away from its learned climatology.",
            "bias": f"{label} jumped to a persistently offset level (calibration/siting fault).",
            "stuck": f"{label} is frozen — unchanged readings for the whole window (sensor latch-up).",
            "dropout": f"{label} observations are missing/zero-filled (telemetry failure).",
            "noise_burst": f"{label} variance is far above its own seasonal norm (electrical/mechanical noise).",
            "clipping": f"{label} is saturating at a repeated ceiling (range clipping).",
            "scale_error": f"{label} amplitude scaled by an abnormal factor (wrong calibration curve).",
            "sensor_swap": f"{label} mirrors another channel's dynamics (wiring/configuration error).",
        }
        parts.append(sentences.get(ft, "Abnormal multivariate pattern."))
        if attribution.get("channels"):
            top = attribution["channels"][0]
            parts.append(f"Dominant channel: {top['label']} "
                         f"(mean |z|={top['mean_abs_z']:.1f}, {top['share']:.0%} of anomaly mass).")
        if qc_hits:
            rules = ", ".join(f"{k}×{v}" for k, v in sorted(qc_hits.items(),
                                                            key=lambda t: -t[1])[:3])
            parts.append(f"Physics QC rules fired: {rules}.")
        parts.append(f"Confidence {conf:.0%}.")
        return " ".join(parts)


# --------------------------------------------------------- IF global importance
def if_permutation_importance(ifdet, X_sample: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permutation importance of IF features using the anomaly score itself as
    target (unsupervised sensibility analysis)."""
    rng = np.random.default_rng(seed)
    base = ifdet.score(X_sample)
    rows = []
    X = X_sample.to_numpy(dtype=np.float32, copy=True)
    for j, c in enumerate(X_sample.columns):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        delta = float(np.mean(np.abs(ifdet.score(pd.DataFrame(Xp, columns=X_sample.columns)) - base)))
        rows.append({"feature": c, "importance": delta})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def try_shap_importance(ifdet, X_sample: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import shap  # noqa
        expl = shap.TreeExplainer(ifdet.model)
        vals = expl.shap_values(X_sample)
        imp = np.abs(vals).mean(axis=0)
        return pd.DataFrame({"feature": X_sample.columns,
                             "importance": imp}).sort_values("importance",
                                                              ascending=False)
    except Exception:
        return None
