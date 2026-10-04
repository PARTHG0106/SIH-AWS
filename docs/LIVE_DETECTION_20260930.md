# Incremental native-minute inference

`awsad.live_detector.LiveMinuteDetector` consumes original timestamped T/P/RH
observations one at a time or in packets. It predicts with the frozen
`artifacts_minute_20260928` models and thresholds. It never reads a saved score to
produce its output. The demonstration is **historical replay through a live
inference engine**, not a connected station feed.

## Contract

- One observation requires `observation_id`, `station_id`, `source`, a UTC-aware
  minute-aligned `timestamp`, and the three nullable measurement channels.
  Field-level source and QC metadata pass through unchanged. QC is not a predictor.
- Predictions use exact observed timestamps at the reviewed lags. No filling,
  resampling or inferred observation is performed. Missing context makes its
  prediction unavailable. Gaps reset abrupt/flatline/sustained continuity; a
  missing channel resets its flatline and invalidates affected residual windows.
- A packet contains at most 360 observations. All input identities/order are
  validated before any packet row is consumed. Old timestamps and repeated IDs
  in retained context are rejected; IDs outside that context are not claimed to
  have an unbounded lifetime deduplication registry.
- Each group retains at most 181 observed input rows, 30 signed residuals per
  channel, one current flatline counter per channel, and a configured finite
  health window. Station/source groups never share observation history.
- Physical and reporting-domain notices are separate from the frozen five-signal
  decision. RH above the conventional 0–100% reporting range can reflect
  supersaturation or reporting effects; it is not declared physically impossible.
  Negative RH, temperature below absolute zero, and nonpositive atmospheric
  station-pressure reports need unit/source review. These notices do not identify
  a hardware cause.
- Signal ratios express threshold exceedance, not confidence. Suggestions such
  as abrupt step, prolonged repetition and persistent forecast mismatch have
  `confidence: null`. Severity is an operational review rule, not fault risk.
- An optional callable `pattern_predictor(raw_history, scored_record)` may add
  `pattern_evidence`. The synthetic pattern model's probabilities retain their
  separate scenario-calibration semantics and do not change frozen scores.

## HTTP integration

Register `make_live_routes(artifact_dir, pattern_predictor=optional_callable)`
from `app.live_api` before the static mount.

| Request | Payload/result |
|---|---|
| `POST /api/live/sessions` | `{group, expected_cadence_minutes: 1}` → `{session_id, snapshot}` |
| `POST /api/live/sessions/{id}/observations` | `{observations: [...]}` → `{results, processing_ms, snapshot}` |
| `GET /api/live/sessions/{id}` | `{session_id, snapshot}` with latest scores and health |
| `POST /api/live/sessions/{id}/advance` | `{timestamp}` → separate data-availability notices |
| `DELETE /api/live/sessions/{id}` | Deletes ephemeral state |
| `GET /api/live/replay?group=...&start=...&end=...` | Original input fields only; start inclusive, end exclusive, maximum six hours |

Sessions are restricted to their declared station/source, capped at 16, and
expire after one idle hour. Request bodies are capped at 2 MB and nonfinite JSON
numbers are rejected. Source/model paths cannot be supplied through the HTTP
request. This local prototype has no durable queue, authentication, instrument
transport, internal scheduler, or multi-process shared session store.

The client drives `advance` with current event time. Without explicitly declared
one-minute cadence, absent slots remain unknown. With cadence, the heartbeat
reports overdue slots even when no new observation arrives. It adds no input
rows. A delayed original report can subsequently resolve an overdue slot.

Cold-start exact context becomes available after 61 minutes for the one-minute
forecast and 120 minutes for the hour forecast; the first complete 30-minute
sustained window is available after 149 minutes. Abrupt-change and flatline
signals can score sooner. Gaps and missing values can extend warmup.

## Health semantics

Health computes candidate alerts divided by **scored** rows for each channel.
Unscored rows cannot dilute that rate. No scored rows means `unknown`, with a null
rate. Trend compares the two halves of elapsed timestamp time, not row count.
Received missing-value fractions and unreported cadence slots are separate.
`healthy` is the legacy name of the low-candidate band only. All bands are review
triage; hardware status remains unknown and no remaining-life estimate is made.

Each channel and station also returns `maintenance_advisory`: evidence,
inspection action, and a policy priority (`review_now`, `next_service`, or
continued observation). Trend-based service triage requires at least 30 scored
readings across one hour; coverage limits and thresholds are included in the
response. Rising or persistent candidate activity can prioritize inspection.
Missing/no-score data instead prompts a feed/logger check. These are transparent
service policies, not a prediction of hardware failure or days remaining.

## Reproducible verification

Run `.venv/Scripts/python.exe scripts/verify_live_detector.py --out NEW_DIRECTORY`.
The script reads original daily SURFRAD files through `NativeSurfradArchive`,
which verifies their acquisition hashes and reconstructs native rows. It feeds
them individually, compares batch predictions/signals/decisions, and exports
new results with original provenance. Neither observations nor labels are
generated or modified.

The default deterministic October prefix was already part of prior development;
this is a compatibility and local-latency check, not an untouched evaluation or
hardware-fault accuracy measurement. Results are in
`artifacts_live_verified_20260930/metrics.json` and `live_scored.jsonl`. Earlier
`artifacts_live_20260930`, `artifacts_live_final_20260930`, and
`artifacts_live_operational_20260930` measurements are
preserved separately. Timings are local measurements affected by concurrent
workloads, not an isolated hardware benchmark. Tests separately
exercise software-only fixtures for missingness, gaps, prolonged flatlines,
station isolation, malformed packets, bounded state, health and HTTP behavior.
