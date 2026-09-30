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
from awsad.benchmark.fault_classifier import FaultTyper
from awsad.station_health import station_health

USA_DIR = os.environ.get("SKYGUARD_ARTIFACTS", str(ROOT / "artifacts_minute_20260928"))
INDIA_DIR = os.environ.get("SKYGUARD_INDIAN_DEMO", str(ROOT / "data" / "indian_demo_20260929"))
DIST = ROOT / "frontend" / "dist"
BENCH = Path(USA_DIR) / "injection_benchmark"
_typer_cache: dict = {}


def _fault_typer():
    if "t" not in _typer_cache:
        path = BENCH / "fault_classifier.joblib"
        try:
            _typer_cache["t"] = FaultTyper.load(path) if path.exists() else None
        except Exception:
            _typer_cache["t"] = None
    return _typer_cache["t"]

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
    typer = _fault_typer()
    ftypes = fconf = None
    if typer is not None:
        try:
            ftypes, fconf = typer.classify(frame)
        except Exception:
            ftypes = fconf = None
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
        if ftypes is not None:
            record["fault_type"] = str(ftypes[idx])
            record["type_confidence"] = _num(fconf[idx])
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
    payload = {"available": BENCH.exists()}
    for name in ("metrics", "classifier"):
        path = BENCH / f"{name}.json"
        payload[name] = json.loads(path.read_text()) if path.exists() else None
    return JSONResponse(payload)


routes = [
    Route("/api/health", lambda request: JSONResponse({"ok": True})),
    Route("/api/benchmark", benchmark),
    Route("/api/usa/catalog", usa_catalog),
    Route("/api/usa/events", usa_events),
    Route("/api/usa/window", usa_window),
    Route("/api/usa/health", usa_health),
    Route("/api/india/catalog", india_catalog),
    Route("/api/india/scenario", india_scenario),
    Route("/api/india/scenario.csv", india_scenario_csv),
]
if DIST.is_dir():
    routes.append(Mount("/", app=StaticFiles(directory=str(DIST), html=True), name="spa"))

app = Starlette(routes=routes, middleware=[
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])])
