"""HTTP protocol invariants using test-only observations and persistence models."""
import asyncio
import json as json_module
from types import SimpleNamespace
from urllib.parse import urlencode

import pandas as pd
from starlette.applications import Starlette

from app.live_api import make_live_routes
from awsad.live_detector import json_safe
from test_live_detector import frame_fixture, make_detector


class ASGIClient:
    """Exercise the actual ASGI protocol without adding a test HTTP dependency."""
    def __init__(self, app):
        self.app = app

    def request(self, method, path, *, json=None, content=None, params=None, headers=None):
        body = json_module.dumps(json).encode() if json is not None else (content or "").encode()
        messages = []

        async def run():
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            async def send(message):
                messages.append(message)

            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                     "http_version": "1.1", "method": method, "scheme": "http", "path": path,
                     "raw_path": path.encode(), "root_path": "", "query_string": urlencode(params or {}).encode(),
                     "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
                     "client": ("test", 1), "server": ("test", 80)}
            await self.app(scope, receive, send)

        asyncio.run(run())
        status = next(message["status"] for message in messages if message["type"] == "http.response.start")
        response = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
        return SimpleNamespace(status_code=status, json=lambda: json_module.loads(response))

    def get(self, path, **kwargs): return self.request("GET", path, **kwargs)
    def post(self, path, **kwargs): return self.request("POST", path, **kwargs)
    def delete(self, path, **kwargs): return self.request("DELETE", path, **kwargs)


def client_factory(**kwargs):
    return ASGIClient(Starlette(routes=make_live_routes("unused", detector_factory=make_detector, **kwargs)))


def test_session_lifecycle_wrong_group_and_atomic_bad_packet():
    client = client_factory()
    created = client.post("/api/live/sessions", json={"group": "fixture|software_test", "expected_cadence_minutes": 1})
    assert created.status_code == 201
    sid = created.json()["session_id"]
    url = f"/api/live/sessions/{sid}"
    rows = json_safe(frame_fixture(3).to_dict("records"))
    scored = client.post(url + "/observations", json={"observations": rows[:1]})
    assert scored.status_code == 200 and scored.json()["results"][0]["hardware_fault_status"] == "unknown"
    bad = [{**rows[1]}, {**rows[2], "source": "other"}]
    assert client.post(url + "/observations", json={"observations": bad}).status_code == 400
    snapshot = client.get(url).json()["snapshot"]
    assert snapshot["groups"]["fixture|software_test"]["rows_seen"] == 1
    heartbeat = client.post(url + "/advance", json={"timestamp": "2024-08-01T00:03:00Z"})
    assert heartbeat.json()["availability"][0]["overdue_slots"] == 3
    assert client.delete(url).status_code == 200
    assert client.get(url).status_code == 404


def test_nonfinite_json_body_rejected_and_idle_sessions_expire():
    now = [0.]
    client = client_factory(clock=lambda: now[0])
    sid = client.post("/api/live/sessions", json={"group": "fixture|software_test"}).json()["session_id"]
    assert client.post(f"/api/live/sessions/{sid}/observations", content='{"observations": [NaN]}',
                       headers={"Content-Type": "application/json"}).status_code == 400
    assert client.post(f"/api/live/sessions/{sid}/observations", content='{"observations": [1e999]}',
                       headers={"Content-Type": "application/json"}).status_code == 400
    now[0] = 3601
    assert client.get(f"/api/live/sessions/{sid}").status_code == 404


def test_replay_returns_original_inputs_and_never_precomputed_scores():
    frame = frame_fixture(5)
    frame["anomaly_score"] = 9999.
    frame["reason_codes"] = "do_not_copy"
    frame["temperature_c__prediction"] = 9999.
    frame["raw_file_sha256"] = "original-hash"
    client = client_factory(replay_loader=lambda group, start, end: frame)
    response = client.get("/api/live/replay", params={"group": "fixture|software_test",
                         "start": "2024-08-01T00:00Z", "end": "2024-08-01T00:05Z"})
    assert response.status_code == 200
    rows = response.json()["observations"]
    assert len(rows) == 5
    assert rows[0]["raw_file_sha256"] == "original-hash"
    assert "anomaly_score" not in rows[0] and "temperature_c__prediction" not in rows[0]
    assert rows[0]["temperature_c"] == frame.temperature_c.iloc[0]
    too_long = client.get("/api/live/replay", params={"group": "fixture|software_test",
                         "start": "2024-08-01T00:00Z", "end": "2024-08-01T07:00Z"})
    assert too_long.status_code == 400
