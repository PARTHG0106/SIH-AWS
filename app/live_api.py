"""Bounded local session API for incremental native-minute observation scoring.

This prototype uses in-memory sessions and client-driven event-time heartbeats.
It provides no station transport, durable queue, authentication or scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
import uuid

import pandas as pd
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from awsad.live_detector import LiveMinuteDetector, json_safe
from awsad.minute_detection import CHANNELS
from awsad.demo.operational_feed import prepare_operational_feed

MAX_BODY_BYTES = 2_000_000
MAX_SESSIONS = 16
SESSION_TTL_SECONDS = 3600


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON numeric constant {value} is not accepted; use null")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number is not accepted; use null")
    return number


async def _body(request):
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_BODY_BYTES:
            raise ValueError("request body exceeds 2 MB")
    try:
        result = json.loads(content, parse_constant=_invalid_constant, parse_float=_finite_float)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("request body must be valid finite JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("request body must be a JSON object")
    return result


@dataclass
class _Session:
    detector: LiveMinuteDetector
    last_access: float


def make_live_routes(artifact_dir: str | Path, replay_loader=None, *, pattern_predictor=None,
                     detector_factory=None, clock=time.monotonic):
    """Build routes; optional factories make protocol tests independent of artifacts.

    ``replay_loader(group, start, end)`` returns original observations in a frame.
    Fields are projected to observation/provenance fields before serialization.
    """
    sessions: dict[str, _Session] = {}
    guard = threading.RLock()
    prototype = None
    replay_bundle = None

    def expire():
        now = clock()
        expired = [sid for sid, session in sessions.items() if now - session.last_access > SESSION_TTL_SECONDS]
        for sid in expired:
            del sessions[sid]

    def lookup(session_id):
        with guard:
            expire()
            if session_id not in sessions:
                raise KeyError("unknown or expired session")
            session = sessions[session_id]
            session.last_access = clock()
            return session.detector

    def create(body):
        nonlocal prototype
        if set(body) - {"group", "expected_cadence_minutes", "health_window_rows"}:
            raise ValueError("unknown session configuration field")
        group = body.get("group")
        if not isinstance(group, str) or len(group) > 513 or group.count("|") != 1 or not all(group.split("|")):
            raise ValueError("one station|source group is required per session")
        health_window = body.get("health_window_rows", 1440)
        if isinstance(health_window, bool) or not isinstance(health_window, int):
            raise ValueError("health_window_rows must be an integer")
        kwargs = {"allowed_group": group, "max_groups": 1,
                  "expected_cadence_minutes": body.get("expected_cadence_minutes"),
                  "health_window_rows": health_window, "pattern_predictor": pattern_predictor}
        with guard:
            expire()
            if len(sessions) >= MAX_SESSIONS:
                raise ValueError("session capacity reached; delete a session or wait for expiry")
            if detector_factory is not None:
                detector = detector_factory(**kwargs)
            else:
                if prototype is None:
                    prototype = LiveMinuteDetector.from_artifacts(artifact_dir)
                detector = LiveMinuteDetector(prototype.models, prototype.thresholds, prototype.config,
                                              model_info=prototype.model_info, **kwargs)
            session_id = uuid.uuid4().hex
            sessions[session_id] = _Session(detector, clock())
            return {"session_id": session_id, "snapshot": detector.snapshot(),
                    "expires_after_idle_seconds": SESSION_TTL_SECONDS}

    async def create_session(request):
        try:
            return JSONResponse(await run_in_threadpool(create, await _body(request)), status_code=201)
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def observations(request):
        try:
            body = await _body(request)
            detector = lookup(request.path_params["session_id"])
            start = time.perf_counter()
            results = await run_in_threadpool(detector.ingest_many, body.get("observations"))
            elapsed = (time.perf_counter() - start) * 1000
            return JSONResponse({"session_id": request.path_params["session_id"], "results": results,
                                 "processing_ms": elapsed, "snapshot": detector.snapshot()})
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def get_session(request):
        try:
            detector = lookup(request.path_params["session_id"])
            return JSONResponse({"session_id": request.path_params["session_id"], "snapshot": detector.snapshot()})
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    async def advance(request):
        try:
            body = await _body(request)
            detector = lookup(request.path_params["session_id"])
            notices = detector.advance(body.get("timestamp"))
            return JSONResponse({"session_id": request.path_params["session_id"], "availability": notices,
                                 "snapshot": detector.snapshot()})
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def delete_session(request):
        with guard:
            sessions.pop(request.path_params["session_id"], None)
        return JSONResponse({"deleted": True})

    def load_replay(query):
        nonlocal replay_bundle
        group = query.get("group")
        if not isinstance(group, str) or len(group) > 513 or group.count("|") != 1:
            raise ValueError("a station|source group is required")
        start, end = pd.Timestamp(query.get("start")), pd.Timestamp(query.get("end"))
        if pd.isna(start) or pd.isna(end) or start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware timestamps")
        if not pd.Timedelta(0) < end - start <= pd.Timedelta(hours=6):
            raise ValueError("replay window must be positive and at most 6 hours")
        if replay_loader is not None:
            frame = replay_loader(group, start, end)
        else:
            from app.real_dashboard import load_bundle, read_observations
            if replay_bundle is None:
                replay_bundle = load_bundle(artifact_dir)
            if group not in set(replay_bundle["catalog"]["group"]):
                raise ValueError("unknown replay station group")
            frame = read_observations(replay_bundle, group, start, end)
        # Only original observation and traceability fields leave this endpoint.
        from app.real_dashboard import OBSERVED_BASE_COLUMNS, OBSERVED_FIELD_SUFFIXES
        keep = {"raw_file_sha256", "raw_file", "raw_file_name", "raw_row_number", *OBSERVED_BASE_COLUMNS}
        keep.update(c for c in frame if any(c == channel + "__" + suffix
                    for channel in CHANNELS for suffix in OBSERVED_FIELD_SUFFIXES))
        frame = frame.loc[(frame.timestamp >= start) & (frame.timestamp < end),
                          [c for c in frame if c in keep]].copy()
        if len(frame) > 360:
            raise ValueError("replay contains more than 360 original rows")
        return {"group": group, "start": start.isoformat(), "end": end.isoformat(), "rows": len(frame),
                "observations": json_safe(frame.to_dict("records")),
                "warmup_minutes": {"forecast_1m": 61, "forecast_60m": 120, "sustained": 149},
                "semantics": "Unchanged archived observations for incremental replay; no precomputed detector scores included."}

    async def replay(request):
        try:
            return JSONResponse(await run_in_threadpool(load_replay, request.query_params))
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    def load_scenario(query):
        # Never open the reserved February final-test source for a demonstration.
        start, end = pd.Timestamp(query.get("start")), pd.Timestamp(query.get("end"))
        if pd.isna(start) or pd.isna(end) or start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware timestamps")
        if start < pd.Timestamp("2025-03-01T00:00:00Z") and end > pd.Timestamp("2025-02-01T00:00:00Z"):
            raise ValueError("February2025 final-test originals are not available for demonstrations")
        original = load_replay(query)
        return prepare_operational_feed(pd.DataFrame(original["observations"]),
            scenario=query.get("scenario", "spike"), channel=query.get("channel", "temperature_c"),
            seed=int(query.get("seed", "26073")))

    async def scenario(request):
        try:
            return JSONResponse(await run_in_threadpool(load_scenario, request.query_params))
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    return [Route("/api/live/sessions", create_session, methods=["POST"]),
            Route("/api/live/sessions/{session_id}", get_session, methods=["GET"]),
            Route("/api/live/sessions/{session_id}", delete_session, methods=["DELETE"]),
            Route("/api/live/sessions/{session_id}/observations", observations, methods=["POST"]),
            Route("/api/live/sessions/{session_id}/advance", advance, methods=["POST"]),
            Route("/api/live/replay", replay, methods=["GET"]),
            Route("/api/live/scenario", scenario, methods=["GET"])]
