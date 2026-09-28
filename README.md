# SkyGuard AI 🛰️ — SIH26073

> **Real-data requirement (2026-09-23):** the user requires original real
> observations and evidence-backed labels. The legacy dataset builders below
> inject faults and interpolate gaps; the current processed data, model results
> and simulated demos are **not approved for a real-only hackathon claim**.
> Read [the real-data audit and acquisition plan](docs/REAL_DATA_RESEARCH.md)
> before building, training or presenting results. The migration is unfinished.
> `python scripts/audit_real_data.py` reproduces the local evidence inventory.

**AI/ML-based intelligent anomaly detection for Automatic Weather Stations (AWS)**
Ministry of Earth Sciences · India Meteorological Department · Disaster Management

SkyGuard AI watches the three AWS measurement channels mandated by the problem
statement — **temperature (°C), pressure (hPa), relative humidity (%)** — and
detects sensor faults, spikes, frozen values, drift, dropouts, clipping, scaling
and wiring errors in near-real time, with explainable root-cause reasoning,
confidence scores, corrected-value estimates and a live dashboard.

---

## Why it's different

| Gap in status quo | SkyGuard AI |
|---|---|
| Threshold QC alarms on real storms (false alarms) | Learns each station's climatology; season-aware residuals — real weather stays quiet |
| Spikes caught, slow faults missed | 7 complementary detectors fuse: physics QC + robust per-channel statistics + Isolation Forest + 2× LSTM autoencoders + Transformer AE + LSTM forecaster + spatial buddy-check |
| One threshold for the whole network | Per-station normalisation + per-station supervised thresholds (validation split), EVT/POT fallback |
| "Anomaly" with no context | Root-cause classifier (9 fault types) + per-channel attribution + human-readable reason strings |
| Dead sensors discovered late | Sensor-health index with maintenance forecast (degradation trend) |
| Cloud-only | Bounded-state QC/statistical streaming path plus quantised deep models for edge hubs; benchmark on the actual target device |
| No labeled AWS fault data exists | Verifiable fault-injection engine on real station data + NAB external benchmark + holdout-station generalisation |

## Verified data foundation (no synthetic-only training)

| Source | What | Verified access |
|---|---|---|
| NOAA ISD (global-hourly) | **~425 Indian surface-station records × 2013–2025 · 4.3 GB raw observations** | AWS-compatible proxy; NOAA metadata does not prove every record is an IMD AWS installation |
| Open-Meteo Archive | co-located reanalysis reference for 21 anchor stations | 21 local raw API exports verified; not presented as sensor truth |
| Numenta NAB | external labeled streaming-method benchmark | 58 local series + MIT license verified |
| NCPOR (MoES) AWS | Maitri/Bharati Antarctic IMD AWS (hourly T/P/RH/wind) | direct link, CAPTCHA-gated → manual drop-in |
| IMD Data Supply Portal | official Indian AWS hourly logs | registration/fee — documented fallback |
| NASA SMAP/MSL (telemanom) | optional spacecraft-telemetry transfer benchmark | not bundled; add separately only if required |

**Kaggle uses a strict two-stage flow:**
1. `krishnagupta02468/skyguard-sih26073` is a small source-only bootstrap
   dataset. It intentionally contains no processed Parquets.
2. `krishnagupta02468/skyguard-ai-data-builder-sih26073` is the online,
   CPU-only notebook that downloads, validates, and emits the authoritative
   processed bundle. Attach only that saved notebook output to GPU training.

## Quick start

```bash
# 1) environment
python -m venv .venv && .venv\Scripts\activate        # or source .venv/bin/activate
pip install -r requirements.txt

# 2) data (for the full corpus, prefer the online Kaggle builder; local fallback:)
python src/awsad/data/download_noaa_isd.py --all-india
python src/awsad/data/download_openmeteo.py
git clone --depth 1 https://github.com/numenta/NAB.git data/raw/nab/NAB

# 3) build the labeled dataset (cleaning + injection of 9 fault types)
python scripts/prepare_dataset.py

# 4) train (CPU quick path shown; full GPU training on Kaggle)
python scripts/run_local_train.py --no-lstm            # classical stack
python scripts/run_local_train.py                      # + deep heads (long on CPU)

# 5) artefacts land in artifacts/ → run the dashboard
streamlit run app/streamlit_app.py
```

### Kaggle training (recommended, offline-safe)

`kaggle/aws_anomaly_training.ipynb` runs everything offline on Kaggle GPU.
First package and upload the source bootstrap:

```bash
python scripts/prepare_kaggle_upload.py   # source-only bootstrap ZIP
```

Run `kaggle_builder/aws_data_builder.ipynb` with Internet on and CPU only,
save its outputs, then attach only that output to the offline GPU notebook.
See `docs/KAGGLE_GUIDE.md` for the exact UI sequence.

### Inference on a new CSV

```bash
python scripts/run_inference.py --input my_station.csv --artifacts artifacts_smoke \
    --out out_my_station
# -> scored.csv (per-observation anomaly scores), alerts.csv (explained events)
```

## Historical development snapshot — Kaggle GPU run (v3, RTX Pro 6000, 129 min)

The numbers below are retained for debugging only. They were produced before the
duplicate-report parser correction and must not be used as the SIH headline
result. Rebuild the online bundle and use its new `artifacts/metrics.json`.

424 station-groups, 2,930 injected test events, all 7 detectors trained:

- **Per-fault event recall:** bias 0.875 · drift 0.829 · noise_burst 0.914 ·
  stuck 0.727 · spike 0.682 · sensor_swap 0.735 · dropout 0.581 ·
  scale_error 0.473 · clipping 0.415
- **Holdout stations** (excluded from ALL fitting): pointF1 **0.437**,
  point-adjusted F1 **0.590**, recall 0.557
- **NAB external benchmark:** ambient-temperature-failure F1 0.69,
  machine-temperature-failure 0.98, nyc-taxi 0.76
- Deep-head convergence logs: LSTM-AE(24h) val 0.114, LSTM-AE(168h) 0.158,
  Transformer-AE 0.096, forecaster 0.081

> Note: v3's report contains a cosmetic defect — a non-finite score poisoned the
> pooled-threshold row and the LOO table (all-zero). That path is hardened in
> dataset v5; single-component and per-fault rows from the v3 run are unaffected.

## Evaluation contract

The checked-in `artifacts_full_classical/` and `artifacts_deep_smoke/` predate the
final leakage/alignment audit and are retained only as development snapshots.
Do not cite their metrics in the SIH submission. Re-run the generated Kaggle
notebook and use its new `artifacts/metrics.json` as the single source of truth.

The corrected protocol reports strict point-wise F1, point-adjusted F1 (with its
known inflation caveat), range-wise F1, PR-AUC, ROC-AUC, detection latency,
per-fault event recall, leave-one-component-out ablation, and a station holdout
score. The station-level holdout set is selected deterministically from the
stations present in the built corpus (20% by default, while retaining the
preferred legacy holdouts when they are available); the exact list and group
coverage are recorded in `data/processed/splits.json`. Every matching
station|source group is excluded from global IF/deep fitting, ensemble-weight
selection, and supervised thresholds. Holdout station-local
climatology/scaling and POT thresholds use clean history only, which is
reported as unsupervised adaptation rather than zero-history cold start.

See `docs/VERIFICATION_REPORT.md` for exact local source evidence and label
limitations, and `docs/DATASET_AUDIT.md` for source-selection, provenance, and
licensing decisions.

## Repo map

```
configs/config.yaml         all knobs (channels, models, injection, thresholds)
src/awsad/                  the library (single source of truth)
  data/                     downloaders + parsers (NOAA ISD, Open-Meteo, NCPOR layout)
  preprocessing/            physics QC · climatology · feature engineering · fault injection
  models/                   qc_rules · statistical · isolation_forest · lstm_autoencoder
                            · transformer_ae · lstm_forecaster · spatial · ensemble
  evaluation/               pointwise/point-adjust/range-wise metrics · POT · NAB
  explain.py · correction.py · health.py · streaming.py
  train_pipeline.py         one pipeline for local + Kaggle
scripts/                    prepare_dataset · run_local_train · prepare_kaggle_upload
                            · run_inference · benchmark_latency · export_edge
kaggle/                     notebook builder + generated offline training notebook
app/streamlit_app.py        dashboard (network map · live sim · alerts · perf · edge)
docs/                       sources · verification · architecture · Kaggle · licenses
tests/test_smoke.py         offline unit tests
```

## Citation / methods

Core techniques with verified references: Telemanom LSTM+NDT (Hundman 2018),
Isolation Forest (Liu 2008), Anomaly-Transformer (Xu 2022, 2110.02642),
TranAD (Tuli 2022, 2201.07284), POT/EVT thresholding (Siffer 2017),
Tatbul range-based metrics (NeurIPS 2018), Kim et al. evaluation rigor (AAAI 2022),
MET Norway titanlib spatial-QC patterns, WMO CIMO sensor-guide fault taxonomy.
