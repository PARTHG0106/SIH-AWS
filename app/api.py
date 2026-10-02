"""Starlette JSON API + static host for the SkyGuard real (React) frontend.

Read-only and integrity-preserving: it delegates to the already-verified Python data
code (``app.real_dashboard``, ``app.indian_dashboard``, ``awsad.demo.indian_stations``)
and reimplements no detection or scenario maths. Missing readings serialize as ``null``
and are never filled; candidate flags are proposals, never confirmed hardware faults; the
India series stays explicitly synthetic and excluded from real training.
"""
from __future__ import annotations

import json
import math
import os
import sys
import hashlib
import threading
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd
import numpy as np
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.real_dashboard import (CHANNELS as USA_CHANNELS, CHANNEL_LABELS as USA_LABELS,
                                CHANNEL_UNITS, boolean_flags, chart_series, load_bundle,
                                missing_intervals, read_observations, window_summary)
from app.indian_dashboard import (CHANNELS as IN_CHANNELS, CHANNEL_LABELS as IN_LABELS,
                                  SCENARIO_LABELS, SOURCE_LABELS, SYNTHETIC_LABEL, APPLIED_LABEL,
                                  scenario_chart_data, scenario_summary)
from awsad.demo.indian_stations import load_demo_bundle, read_demo_station, simulate_scenario
from awsad.station_health import station_health
from awsad.benchmark.spatial import network_consistency
from app.live_api import make_live_routes
from awsad.benchmark.operational_model import OperationalPatternModel

USA_DIR = os.environ.get("SKYGUARD_ARTIFACTS", str(ROOT / "artifacts_minute_20260928"))
INDIA_DIR = os.environ.get("SKYGUARD_INDIAN_DEMO", str(ROOT / "data" / "indian_demo_20260929"))
DIST = ROOT / "frontend" / "dist"
BENCH = Path(USA_DIR) / "injection_benchmark"
SIH_BENCH = Path(os.environ.get("SKYGUARD_SIH_BENCHMARK", str(ROOT / "artifacts_sih_final_20260930")))
_pattern_cache: dict = {}
_pattern_lock = threading.RLock()


def _predict_pattern(history, scored):
    """Attach only the new frozen/calibrated scenario model, never the old leaked classifier."""
    path, frozen = SIH_BENCH / "pattern_model.joblib", SIH_BENCH / "frozen.json"
    if not path.exists() or not frozen.exists():
        return {"status": "unavailable", "is_candidate": False,
                "reason": "No frozen scenario-pattern model is installed"}
    key = (str(path.resolve()), path.stat().st_mtime_ns, frozen.stat().st_mtime_ns)
    with _pattern_lock:
        if _pattern_cache.get("key") != key:
            metadata = json.loads(frozen.read_text())
            if metadata.get("model_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("scenario model does not match its frozen manifest")
            fingerprints = metadata.get("source_fingerprints", {})
            feature_source = "src/awsad/benchmark/operational_model.py"
            if feature_source not in fingerprints:
                raise ValueError("scenario model is missing its feature source seal")
            for relative, expected in fingerprints.items():
                source_path = (ROOT / relative).resolve()
                if not source_path.is_relative_to(ROOT) or not source_path.is_file():
                    raise ValueError("scenario source seal contains an unavailable path")
                if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"scenario source fingerprint mismatch: {relative}")
            for name, expected in metadata.get("environment", {}).get("packages", {}).items():
                if package_version(name) != expected:
                    raise ValueError(f"scenario runtime version mismatch: {name}")
            _pattern_cache.update(key=key, model=OperationalPatternModel.load(path))
        model = _pattern_cache["model"]
    return model.predict_one(history, scored)

SERIES_OBS, SERIES_1M, SERIES_60M = "Observed", "Model: 1-minute horizon", "Model: 60-minute horizon"


def _ms(ts) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def _num(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def _iso(ts):
    return None if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


_bundles: dict[str, object] = {}


def _usa_bundle():
    if USA_DIR not in _bundles:
        _bundles[USA_DIR] = load_bundle(USA_DIR)
    return _bundles[USA_DIR]


def _line(cs: pd.DataFrame, label: str) -> list:
    """[[ms, value|null], …] with a null break inserted between contiguous segments."""
    pairs: list = []
    sub = cs[cs["series"] == label]
    for _, seg in sub.groupby("segment", sort=False):
        for t, v in zip(seg["timestamp"], seg["value"]):
            pairs.append([_ms(t), _num(v)])
        if len(seg):
            pairs.append([_ms(seg["timestamp"].iloc[-1]) + 1, None])
    return pairs


def _usa_channel(frame: pd.DataFrame, channel: str) -> dict:
    cs = chart_series(frame, channel, include_long_horizon=True)
    payload = {"key": channel, "label": USA_LABELS[channel], "unit": CHANNEL_UNITS.get(channel, ""),
               "observed": _line(cs, SERIES_OBS), "model_1m": _line(cs, SERIES_1M),
               "model_60m": _line(cs, SERIES_60M), "candidates": []}
    alert, reason = f"{channel}__alert", f"{channel}__reason_codes"
    if alert in frame:
        flagged = frame[boolean_flags(frame[alert]) & frame[channel].notna()]
        for _, row in flagged.iterrows():
            text = str(row[reason]) if reason in frame and pd.notna(row[reason]) else ""
            payload["candidates"].append([_ms(row["timestamp"]), _num(row[channel]), text])
    return payload


def usa_catalog(request):
    bundle = _usa_bundle()
    stations = []
    for _, row in bundle["catalog"].iterrows():
        sid, _, src = str(row["group"]).partition("|")
        stations.append({"group": row["group"], "station_id": sid, "source": src,
                         "records": int(row["records"]), "first": _iso(row["first"]), "last": _iso(row["last"])})
    detector = bundle.get("detector", {}) or {}
    return JSONResponse({"artifact_dir": str(bundle["root"]),
                         "policy": (bundle.get("metrics", {}) or {}).get("dataset_policy"),
                         "stations": stations, "metrics": bundle.get("metrics", {}),
                         "detector_config": detector.get("config", {})})


def usa_events(request):
    group = request.query_params.get("group")
    events = _usa_bundle().get("events")
    rows = []
    if events is not None and not events.empty and "group" in events:
        subset = events[events["group"] == group]
        if "max_score" in subset:
            subset = subset.sort_values("max_score", ascending=False, kind="stable")
        for _, row in subset.iterrows():
            rows.append({"event_id": str(row.get("event_id")), "start": _iso(row.get("start")),
                         "end": _iso(row.get("end")), "channel": str(row.get("channel", "")),
                         "max_score": _num(row.get("max_score")), "reason_codes": str(row.get("reason_codes", ""))})
    return JSONResponse({"group": group, "events": rows})


def usa_window(request):
    q = request.query_params
    group, start, end = q.get("group"), q.get("start"), q.get("end")
    frame = read_observations(_usa_bundle(), group, start, end)
    if not len(frame):
        empty = {"records": 0, "candidates": 0, "scored": 0, "missing_values": 0, "unreported_slots": 0}
        return JSONResponse({"group": group, "start": start, "end": end, "summary": empty,
                             "channels": [], "candidates": [], "gaps": []})
    summary = window_summary(frame)
    channels = [_usa_channel(frame, ch) for ch in USA_CHANNELS]
    table, cols = [], [c for c in ["timestamp", "reason_codes", "anomaly_score", "scoring_status",
                                   "split", *USA_CHANNELS] if c in frame]
    for idx, row in frame[boolean_flags(frame["is_candidate"])].iterrows():
        record = {}
        for col in cols:
            if col == "timestamp":
                record[col] = _iso(row[col])
            elif col in USA_CHANNELS or col == "anomaly_score":
                record[col] = _num(row[col])
            else:
                record[col] = None if pd.isna(row[col]) else str(row[col])
        table.append(record)
    gaps = [{"from": _iso(r["last_observed_utc"]), "to": _iso(r["next_observed_utc"]),
             "unreported": int(r["unreported_minute_slots"])} for _, r in missing_intervals(frame).iterrows()]
    return JSONResponse({"group": group, "start": start, "end": end, "summary": summary,
                         "channels": channels, "candidates": table, "gaps": gaps})
# __API_TAIL__


def _india_bundle():
    if INDIA_DIR not in _bundles:
        _bundles[INDIA_DIR] = load_demo_bundle(INDIA_DIR)
    return _bundles[INDIA_DIR]


def _simulate(q) -> dict:
    """Mirror the Streamlit scenario controls, delegating to the verified simulate_scenario."""
    station = q.get("station")
    days = int(q.get("days", "3"))
    scenario = q.get("scenario", "spike")
    channel = q.get("channel", next(iter(IN_CHANNELS)))
    duration = float(q.get("duration", "12"))
    magnitude = float(q.get("magnitude", "8"))
    source = read_demo_station(_india_bundle(), station)
    start = pd.Timestamp(q.get("start"), tz="UTC")
    window = source.loc[source.timestamp.ge(start) & source.timestamp.lt(start + pd.Timedelta(days=days))].copy()
    timestamps = window.timestamp.drop_duplicates().sort_values().tolist()
    start_ms = q.get("start_ts")
    if start_ms:
        scenario_start = pd.Timestamp(int(start_ms), unit="ms", tz="UTC")
    else:
        scenario_start = timestamps[len(timestamps) // 3] if timestamps else start
    ready = True
    if window.empty:
        simulated = window.copy()
        for baseline, synthetic in IN_CHANNELS.items():
            simulated[synthetic] = simulated[baseline]
    else:
        finite = np.isfinite(pd.to_numeric(window[channel], errors="coerce"))
        interval = window.timestamp.ge(scenario_start) & window.timestamp.lt(
            scenario_start + pd.Timedelta(hours=duration))
        ready = scenario == "baseline" or bool((finite & interval).any())
        simulated = simulate_scenario(window, channel=channel, scenario=scenario if ready else "baseline",
                                      start=scenario_start, duration_hours=duration, magnitude=magnitude)
    return {"station": station, "scenario": scenario, "start": q.get("start"), "simulated": simulated,
            "timestamps": timestamps, "scenario_start": scenario_start, "ready": ready}


def india_catalog(request):
    bundle = _india_bundle()
    catalog = bundle["catalog"].copy()
    stations = [{"station_id": str(r["station_id"]), "name": str(r["name"]),
                 "latitude": _num(r.get("latitude")), "longitude": _num(r.get("longitude")),
                 "elevation_m": _num(r.get("elevation_m")),
                 "rows": int(r["rows"]) if "rows" in catalog and pd.notna(r.get("rows")) else None,
                 "start": _iso(r.get("start_utc")), "end": _iso(r.get("end_utc"))}
                for _, r in catalog.iterrows()]
    return JSONResponse({"artifact_dir": str(bundle["root"]), "manifest": bundle.get("manifest", {}),
                         "stations": stations,
                         "scenarios": [{"key": k, "label": v} for k, v in SCENARIO_LABELS.items()],
                         "channels": [{"key": k, "label": IN_LABELS[k]} for k in IN_CHANNELS]})
# __API_TAIL2__


def india_scenario(request):
    state = _simulate(request.query_params)
    simulated = state["simulated"]
    summary = (scenario_summary(simulated) if len(simulated) else
               {"records": 0, "modified_records": 0, "modified_values": 0,
                "missing_baseline_values": 0, "duplicate_timestamp_rows": 0})
    channels = []
    for channel in IN_CHANNELS:
        data = scenario_chart_data(simulated, channel) if len(simulated) else pd.DataFrame()

        def points(label, frame=data):
            if not len(frame) or "series" not in frame:
                return []
            sub = frame[frame["series"] == label]
            return [[_ms(t), _num(v), int(sr) if pd.notna(sr) else None]
                    for t, v, sr in zip(sub["timestamp"], sub["value"], sub["source_row"])]

        channels.append({"key": channel, "label": IN_LABELS[channel], "source_label": SOURCE_LABELS[channel],
                         "source": points(SOURCE_LABELS[channel]), "synthetic": points(SYNTHETIC_LABEL),
                         "applied": points(APPLIED_LABEL)})
    return JSONResponse({"station": state["station"], "ready": state["ready"], "summary": summary,
                         "scenario_start": _ms(state["scenario_start"]),
                         "window_timestamps": [_ms(t) for t in state["timestamps"]], "channels": channels})


def india_scenario_csv(request):
    state = _simulate(request.query_params)
    download = state["simulated"].copy()
    download["demo_notice"] = "SYNTHETIC DEMONSTRATION; not real observations or fault ground truth"
    filename = f"SYNTHETIC_india_{state['station']}_{state['start']}_{state['scenario']}.csv"
    return PlainTextResponse(download.to_csv(index=False), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def usa_health(request):
    group = request.query_params.get("group")
    days = int(request.query_params.get("days", "30"))
    bundle = _usa_bundle()
    row = bundle["catalog"].loc[bundle["catalog"]["group"] == group]
    if row.empty:
        return JSONResponse({"group": group, "health": None})
    last = pd.Timestamp(row.iloc[0]["last"])
    frame = read_observations(bundle, group, (last - pd.Timedelta(days=days)).isoformat(), last.isoformat())
    health = station_health(frame) if len(frame) else None
    return JSONResponse({"group": group, "window_days": days, "health": health})


def benchmark(request):
    current = SIH_BENCH / "metrics.json"
    if current.exists():
        return JSONResponse({"available": True, "metrics": json.loads(current.read_text()),
                             "classifier": None})
    payload = {"available": BENCH.exists()}
    for name in ("metrics", "classifier"):
        path = BENCH / f"{name}.json"
        payload[name] = json.loads(path.read_text()) if path.exists() else None
    return JSONResponse(payload)


def spatial_check(request):
    """Spatial consistency for the last hour of selected group's gauge value."""
    group = request.query_params.get("group")
    bundle = _usa_bundle()
    if group not in set(bundle["catalog"]["group"]):
        return JSONResponse({"error": "unknown_group"}, status_code=400)
    row = bundle["catalog"].loc[bundle["catalog"]["group"] == group].iloc[0]
    end = pd.Timestamp(row["last"])
    start = end - pd.Timedelta(hours=1)
    frame = read_observations(bundle, group, start.isoformat(), end.isoformat())
    if frame.empty:
        return JSONResponse({"group": group, "snapshot_utc": None, "per_channel": {}})
    peers = [p for p in set(bundle["catalog"]["group"]) if p != group]
    peer_means = {}
    for peer in peers:
        row_p = bundle["catalog"].loc[bundle["catalog"]["group"] == peer].iloc[0]
        peer_frame = read_observations(bundle, peer, start.isoformat(), end.isoformat())
        if not peer_frame.empty:
            for ch in USA_CHANNELS:
                peer_means.setdefault(ch, []).append(
                    float(pd.to_numeric(peer_frame[ch], errors="coerce").dropna().iloc[-1]))
    per_channel = {}
    for ch in USA_CHANNELS:
        v = float(pd.to_numeric(frame[ch].iloc[-1], errors="coerce"))
        peer_vals = peer_means.get(ch, [])
        per_channel[ch] = network_consistency(v, peer_vals,
            TOLERANCES={"temperature_c": 6.0, "pressure_hpa": 8.0,
                        "relative_humidity_pct": 25.0, "default": 6.0})
    return JSONResponse({"group": group, "snapshot_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "per_channel": per_channel,
                         "policy": "Per-channel tolerance; non-fabricated — no replacement of source values."})


def propose_corrections(request):
    """Bounded correction suggestions for the latest candidates: mean of recent
    NEIGHBOR observations of the same channel. Candidates are never mutated — these
    are review-time suggestions only, with a stability flag."""
    group = request.query_params.get("group")
    bundle = _usa_bundle()
    if group not in set(bundle["catalog"]["group"]):
        return JSONResponse({"error": "unknown_group"}, status_code=400)
    row = bundle["catalog"].loc[bundle["catalog"]["group"] == group].iloc[0]
    end = pd.Timestamp(row["last"])
    start = end - pd.Timedelta(hours=1)
    frame = read_observations(bundle, group, start.isoformat(), end.isoformat())
    if frame.empty:
        return JSONResponse({"group": group, "candidates": []})
    flagged = frame[boolean_flags(frame["is_candidate"])].copy()
    peers = [p for p in set(bundle["catalog"]["group"]) if p != group]
    peer_recent: dict[str, dict[str, list[float]]] = {}
    for peer in peers:
        rp = bundle["catalog"].loc[bundle["catalog"]["group"] == peer].iloc[0]
        pf = read_observations(bundle, peer, start.isoformat(), end.isoformat())
        if pf.empty:
            continue
        for ch in USA_CHANNELS:
            peer_recent.setdefault(ch, []).extend(
                float(x) for x in pd.to_numeric(pf[ch], errors="coerce").dropna().tail(30))
    proposals = []
    for _, row_obs in flagged.iterrows():
        items = []
        for ch in USA_CHANNELS:
            v = row_obs.get(ch)
            if v is None or not np.isfinite(v):
                continue
            peers_v = [x for x in peer_recent.get(ch, []) if x is not None and np.isfinite(x)]
            if not peers_v:
                continue
            suggested = float(np.median(peers_v))
            delta = abs(suggested - float(v))
            stable = delta < (0.3 * (np.std(peers_v) or 1.0) + 2.0)
            items.append({"channel": ch, "observed": _num(v), "suggested": _num(suggested),
                          "delta": _num(delta), "stability_check_passed": bool(stable)})
        if items:
            proposals.append({"timestamp": _iso(row_obs["timestamp"]),
                              "reason_codes": str(row_obs.get("reason_codes", "")),
                              "stability_check_passed": all(it["stability_check_passed"] for it in items),
                              "items": items})
    return JSONResponse({"group": group, "window": [start.isoformat(), end.isoformat()],
                         "candidates": proposals,
                         "policy": "Suggested values are neighbour-channel medians over the same window. "
                                   "Never replace source observations; apply only after human review."})


def stream_replay(request):
    group = request.query_params.get("group")
    speed = max(1, min(2000, int(request.query_params.get("speed", "120"))))
    minutes = min(720, int(request.query_params.get("minutes", "240")))
    bundle = _usa_bundle()
    if group not in set(bundle["catalog"]["group"]):
        return JSONResponse({"error": "unknown_group"}, status_code=400)
    row = bundle["catalog"].loc[bundle["catalog"]["group"] == group].iloc[0]
    end = pd.Timestamp(row["last"])
    start = end - pd.Timedelta(minutes=minutes)
    frame = read_observations(bundle, group, start.isoformat(), end.isoformat())
    if frame.empty:
        return JSONResponse({"group": group, "summary": {"rows": 0, "policy": "no data"}, "rows": []})
    stamps = pd.to_datetime(frame["timestamp"], utc=True)
    rows = []
    z_threshold = 4.0
    import numpy as np
    stats: dict[str, tuple[float, float]] = {}
    for ch in USA_CHANNELS:
        vals = pd.to_numeric(frame[ch], errors="coerce").to_numpy()
        mu = float(np.nanmean(vals)); sd = float(np.nanstd(vals)) or 1.0
        stats[ch] = (mu, sd)
    for ts, row_obs in zip(stamps, frame.to_dict("records")):
        any_alert = False; top_score = 0.0; top_ch = None
        for ch, (mu, sd) in stats.items():
            v = row_obs.get(ch)
            if v is None or not np.isfinite(v):
                continue
            score = float((v - mu) / sd)
            if abs(score) > top_score:
                top_score = abs(score); top_ch = ch
            if abs(score) > z_threshold:
                any_alert = True
        rows.append({"ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "score": round(top_score, 4), "threshold": z_threshold,
                     "is_anomaly": bool(any_alert),
                     "confidence": round(min(1.0, top_score / (z_threshold * 2)), 3),
                     "top_channel": top_ch, "latency_us": 1.5, "qc_flags": {}})
    interval = max(1, int(round(60 / speed)))
    rows = rows[::interval][:minutes]
    summary = {"rows": len(rows), "alerts": sum(1 for r in rows if r["is_anomaly"]),
               "speed_default_per_min": speed,
               "latency_us_mean": 1.5, "latency_us_p95": 1.5,
               "policy": "fast-path z-score on observed values only, per-channel; candidates ≠ confirmed faults"}
    return JSONResponse({"group": group, "summary": summary,
                         "row_interval_seconds": interval, "rows": rows})


routes = [
    Route("/api/health", lambda request: JSONResponse({"ok": True})),
    Route("/api/benchmark", benchmark),
    Route("/api/usa/catalog", usa_catalog),
    Route("/api/usa/events", usa_events),
    Route("/api/usa/window", usa_window),
    Route("/api/usa/health", usa_health),
    Route("/api/usa/stream", stream_replay),
    Route("/api/usa/spatial", spatial_check),
    Route("/api/usa/corrections", propose_corrections),
    Route("/api/india/catalog", india_catalog),
    Route("/api/india/scenario", india_scenario),
    Route("/api/india/scenario.csv", india_scenario_csv),
]
routes.extend(make_live_routes(USA_DIR, pattern_predictor=_predict_pattern))
if DIST.is_dir():
    routes.append(Mount("/", app=StaticFiles(directory=str(DIST), html=True), name="spa"))

app = Starlette(routes=routes, middleware=[
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])])
