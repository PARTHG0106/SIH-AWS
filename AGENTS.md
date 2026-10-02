# AGENTS.md — project knowledge for coding agents

## SIH improvement authorization (2026-09-30)
- The latest user request authorizes researching public implementations and using
  real data or a good data-generation pipeline to meet SIH26073 requirements.
- A separate, explicitly synthetic scenario training/evaluation path is now
  authorized. This supersedes the earlier demo-only restriction for that path.
  Preserve original observations, raw quality flags, provenance and hashes.
- Synthetic targets describe applied software scenarios, never confirmed real
  hardware causes. Unmodified backgrounds mean no injected scenario, not known
  fault-free hardware. Keep real replay metrics and synthetic benchmark metrics
  separate, including all exported artifacts and dashboard claims.
- Split source windows/stations/time before generating scenarios. Keep events
  and overlapping history in one partition. Fit, selection, calibration and final
  evaluation must remain separate; previously examined January 2025 is not an
  untouched selection holdout.
- Research and compare inspected public implementations with citations and
  reproducible measurements; do not claim superiority over uninspected projects.
- Do not revive the legacy dataset builder, which fills observations and
  conflates derived RH/sea-level pressure with measured sensor channels.

## Explicit Indian demo exception (2026-09-29)
- The latest user request authorizes synthetic Indian-station scenarios on a
  separate dashboard page while retaining the US real-observation replay.
- This exception is for a visibly labelled demonstration only. Keep source
  records unchanged and simulated values in separate columns/artifacts. Never
  feed them into real-observation training, calibration, evaluation or claims.
- Use original Indian NOAA ISD reports as the scenario baseline. RH calculated
  from temperature/dew point is derived; SLP is sea-level pressure. Neither
  establishes three independently measured AWS inputs or IMD AWS validation.
- Scenario markers describe deliberate software changes, not model detections
  or confirmed real faults. Actual hardware-fault status remains unknown.
- See `docs/INDIAN_STATION_DEMO.md`; do not revive the legacy injected pipeline.

## Current user requirement — real data only (2026-09-23)
- The user explicitly forbids assumed/manufactured dataset observations and
  labels. This supersedes the legacy injected-data workflow documented below.
- Read `docs/REAL_DATA_RESEARCH.md` before dataset, training or demo work.
- Do not run the legacy synthetic builders or present their artifacts as a
  compliant real-data result. The 2026-09-26 SURFRAD training path is separate
  from the historical ISD/injected pipeline and dashboard artifacts.
- Missing readings stay missing; unknown fault labels stay unknown. Do not
  substitute interpolation, constants, reanalysis or model predictions for
  observations, or label unreviewed rows fault-free.
- Distinguish independently measured RH from RH calculated from T/dew point,
  and station pressure from sea-level pressure. Preserve field-level source,
  timestamp, unit, raw quality/measurement codes and immutable raw-file hashes.
- Provider QC flags are quality evidence, not confirmed hardware root causes.
  Derived physics features and model predictions must remain explicitly
  separate from observed inputs and ground truth.
- Candidate sources require actual provider-document/sample verification;
  source availability, licences, station coverage and sensor types are not to
  be inferred. The 2026-09-23 live-source research was blocked by network
  permissions/approval-review timeouts; do not call those candidates verified.
- Synthetic software-test fixtures may remain in tests only; never put them
  into the dataset, benchmark claims or a purported real-feed demonstration.

## What this is
SkyGuard AI: anomaly detection for Automatic Weather Stations (SIH26073, MoES/IMD).
Detection reads ONLY: temperature_c, pressure_hpa, relative_humidity_pct
(+ physics-derived dewpoint/td_spread/es/vpd/press_tendency_3h). Python 3.14,
package `src/awsad`, tests in `tests/`.

## How to run
- SIH continuation (2026-10-01): completed frozen synthetic evaluation is in
  `artifacts_sih_final_20260930`; see `docs/SIH_FINAL_RESULTS_20261001.md`.
  February 2025 was consumed once. Never retune on it or call it untouched again.
  Learned-model temporal F1 is 0.689; held-out-station F1 is 0.374 with 49.65%
  wholly unmodified background proposals. Reliable station transfer remains
  unestablished; preserve this limitation in dashboard/submission claims.
  `scripts/package_sih_release.py --pattern-dir artifacts_sih_final_20260930`
  selects the completed run; packaging uses the stable `artifacts_sih_20260930`
  alias inside the ZIP. One-click synthetic stream is in the live dashboard.
- Native-minute continuation (2026-09-29): see `docs/MINUTE_DETECTION_20260929.md`.
  `scripts/run_minute_detection.py` uses verified native originals with separate
  fit/selection/calibration before 2024-11; Goodwin Creek is held out. January
  2025 has now been replayed: do not reuse it for further selection while calling
  it untouched. Final artifacts: `artifacts_minute_20260928` and
  `artifacts_minute_fresh_20260928`; fault labels remain unknown.
- Active dashboard: **React + Vite + ECharts SPA** in `frontend/`, served with the
  Starlette API `app/api.py` (reuses the verified data code; missing stays null,
  candidates stay proposals, India stays synthetic). Build + run:
  `npm --prefix frontend run build` then `python scripts/serve_dashboard.py [port]`
  (default 8501). The legacy `streamlit run app/streamlit_app.py` still works but is
  superseded by the SPA. Real minute replay artifacts back both. New Kaggle entry:
  `kaggle/aws_minute_detection.ipynb`, generator `kaggle/build_minute_notebook.py`;
  locally validated, not remotely published/run. Refresh its embedded hashes after
  changing minute source.
- Reviewed-event evaluation: `scripts/evaluate_real_events.py`. Never convert QC
  or detector proposals into confirmed labels. Evidence registry is under
  `docs/research/surfrad_events_20260928/`.
- Real-data path: `scripts/fetch_real_surfrad.py`, then
  `awsad.data.surfrad.build_surfrad_dataset`; strict admission reconstructs every
  retained measurement from immutable originals. See `docs/KAGGLE_GUIDE.md`.
- Real training: `python scripts/run_real_observation_train.py --dataset-dir DIR
  --out NEW_DIR --config configs/real_surfrad.json --selection-only`; use
  `--evaluate-test` only after freezing the configuration.
- Real notebook generators: `kaggle/build_data_builder.py` and
  `kaggle/build_notebook.py`. Refresh fingerprints after editing training source.
- SURFRAD native files publish one-minute sensor averages; the modeling view
  selects actual minute00 records. Missing hours are absent, never filled.
- The commands and scale descriptions below describe the **legacy** workflow.
  Do not run its preparation or demos for the real-data task.
- Env: `.venv` (activated: `pip install -r requirements.txt`, torch CPU wheel works on 3.14)
- **Data scale**: full corpus = ~425 ISD stations × 2013-2025 (4.3 GB raw →
  ~930 MB processed parquets). `run_local_train.py` supports `--max-groups N` and
  the pipeline is float32/per-group (RAM-safe for 30 GB Kaggle).
- On Kaggle: dataset `krishnagupta02468/skyguard-sih26073` is already v3;
  an ONLINE CPU-only builder notebook (`skyguard-ai-data-builder-sih26073`)
  recreates the whole dataset on Kaggle itself.
- Data prep end-to-end: `python scripts/prepare_dataset.py` (raw → processed with injected labels)
- Train (CPU ok, use flag to skip deep): `python scripts/run_local_train.py --no-lstm`
- Full smoke: `python -m pytest tests/ -q`
- Dashboard: `streamlit run app/streamlit_app.py` (point sidebar at artifacts dir)
- Kaggle packaging: `python scripts/prepare_kaggle_upload.py` then see docs/KAGGLE_GUIDE.md

## Hard-won conventions / gotchas
- **Alignment is sacred**: never reorder frames between features and labels;
  scale groups in-place by index (`_scale_all_groups` keeps original order).
- A **station|source "group"** (e.g. `42182099999|noaa_isd`) is the unit of
  climatology/normalisation/threshold. Keep that granularity everywhere.
- NOAA ISD quirks: some station-years lack `AA1`; missing codes like +9999;
  airports report sub-hourly → resample hourly with median (never `asfreq`,
  it silently drops off-grid rows).
- Open-Meteo: occasional HTTP-200 bodies containing error text → validate
  payload before accepting; fallback to default models on retry.
- Smooth scores with `max(ewma(x), x)` — plain EWMA eats 1-point spikes.
- Thresholds: per-group supervised on val split; POT(telem) only as unseen-station
  fallback. Never tune on the test split.
- Local PowerShell: no heredocs (`<<`); use python files for ad-hoc probes.
  Detached `Start-Process` children survive (that's how big downloads finished).
- pandas 3.x in .venv: mind `.fillna(method=...)` removal and `nan` groupby medians.
- Keep judge-facing numbers reproducible: seeds fixed (42/0), results in
  `artifacts*/metrics.json`.
