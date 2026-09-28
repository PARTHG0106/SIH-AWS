# AGENTS.md — project knowledge for coding agents

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
