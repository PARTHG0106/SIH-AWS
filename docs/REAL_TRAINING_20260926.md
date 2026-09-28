# Real-observation training result — 26 September 2026

The new causal forecasting model improves one-hour prediction error on held-out
real observations, with **zero synthetic training observations and zero invented
fault labels**. Both Kaggle notebooks now use the verified SURFRAD path.

## What was acquired and verified

- 2,167 original NOAA SURFRAD daily files, covering available provider-listed
  days for Bondville, Fort Peck and Goodwin Creek during 2023–2024.
- 3,099,744 native published one-minute records preserved in immutable files.
- 51,652 existing minute00 observations selected for modeling; no hourly
  averaging, gap filling, clipping, noise injection or inserted timestamps.
- 23 missing temperature values, 23 missing RH values and 141 missing station
  pressure values remain missing in that selected view.
- Every retained value, raw string, timestamp, code and lineage identifier was
  reconstructed from its original file and compared with the processed table.
- Independently measured RH and station pressure are established by the
  [inspected provider documentation](research/surfrad_20260926/VERIFICATION.md).
  Provider files publish native averages of real sensor samples.

The old injected/derived-RH/sea-level-pressure dataset is not an input to this
experiment. DWD was researched but was not admitted because its exported
correction/measurement evidence did not satisfy the current policy.

## Frozen experiment

The model receives past observed T/P/RH values, changes in those values and
explicit time features. It compares persistence, the previous day's observed
value, robust median change, gradient boosting of levels and gradient boosting
of changes (full or half-strength). The change learner won for all three
channels on validation. Its targets are differences between two actual measured
values; they are not synthetic observations or hardware-fault labels.

Train: Bondville and Fort Peck, January 2023–August 2024. The model-fit sample is
capped at 20,000 accepted actual contexts/targets per channel, with a separate
training tail for residual scales. September–October 2024 is divided into model
selection and threshold-calibration periods. Goodwin Creek never contributes
to fitting, selection, residual scaling or threshold calibration.

The configuration was frozen at **2026-09-26 03:08:10 UTC**, before evaluating
November–December 2024. Selection used no test target values or test QC.
The model uses 80 boosting iterations, 15 leaves, learning rate 0.05, L2=3,
seed 42 and four CPU threads. Group scales and thresholds are indexed by
station|source; unseen stations use an explicit pooled fallback.

## Final local test results

One-hour mean absolute error on identical complete, provider-accepted target
windows. RH error is in percentage points. There are 2,916 eligible test rows
across the two seen stations and 1,458 at the held-out station, over 61 days.

| Channel | Seen-station persistence → model MAE | Error reduction | Held-out-station persistence → model MAE | Error reduction |
|---|---:|---:|---:|---:|
| Temperature, °C | 0.771 → 0.581 | **24.6%** | 0.807 → 0.530 | **34.3%** |
| Station pressure, hPa | 0.472 → 0.316 | **33.0%** | 0.424 → 0.301 | **29.1%** |
| Relative humidity, points | 2.994 → 2.579 | **13.9%** | 3.665 → 2.851 | **22.2%** |

Paired seven-day moving-block bootstrap intervals (2,000 resamples, seed 42)
for percentage error reduction were respectively 20.2–28.2%, 29.0–35.2% and
8.4–17.8% at seen stations; 30.9–37.8%, 24.3–34.0% and 17.8–25.8% at the held-out
station. These resample existing error statistics only; no resampled or
generated observations enter training. The intervals are conditional on this
corpus and fitted model, not a guarantee for other networks.

The final test contains 4,389 original rows; 4,374 can be scored with complete
past context. The aggregate model flags 6 of 2,916 scored seen-station rows and
4 of 1,458 scored held-out-station rows. These are **alert rates**, not verified
false-positive rates. Unavailable contexts remain unscored.

## What these results support

They demonstrate lower real-observation forecasting error, including transfer
to a station excluded from training. Forecast residuals support anomaly scoring,
but there are no independently reviewed hardware-fault labels in this corpus.
No sensor-fault F1, nine-class cause accuracy or correction accuracy is claimed.
The scored test targets have no provider-QC positives, so provider-QC recall
also cannot be estimated here. This is a US SURFRAD result, not IMD AWS validation.

The new artifact format is separate from the historical dashboard's nine-class
model loader. The training notebook exports an explicitly historical real
observation replay with separate predictions and scores.

## Evidence and reproduction

### Completed Kaggle reproduction

Source dataset **version 19** is published and its downloaded manifest matches
the staged source exactly. The online builder **version 18** completed on Kaggle,
acquiring all 2,167 provider-listed daily files and verifying 51,652 selected rows.
Training **version 13** completed with the same frozen configuration, using that
verified builder output directly. Model fitting/scoring took 4.20 seconds on CPU;
this excludes data admission, startup and notebook export time.

Kaggle runtime was Python 3.12.13, pandas 2.3.3, NumPy 2.0.2 and scikit-learn 1.6.1.
Its error reductions were 24.57%, 33.03%, 13.89% for T/P/RH at seen stations, and
34.15%, 28.99%, 22.37% at the held-out station. Small differences from the local
run accompany the recorded runtime versions; no test-driven tuning occurred.

- [Completed training notebook](https://www.kaggle.com/code/krishnagupta02468/skyguard-ai-aws-anomaly-training-sih26073)
- [Completed builder notebook](https://www.kaggle.com/code/krishnagupta02468/skyguard-ai-data-builder-sih26073)
- Cloud artifacts: `artifacts_real_20260926/kaggle_v13/artifacts/`.
- All 148 software tests and 6 subtests passed. Full raw-to-processed admission
  also passed under pandas 2.2.3 and the Kaggle runtime.

**Named-dataset workflow verified:** the user refreshed
`skyguard-sih26073-processed` to **version 9**. Its manifest and complete inventory
of 2,299 files match builder version 18. Training **version 14** then completed
with only that named dataset attached, requested explicitly at version 9.
The source and full raw-to-processed checks passed again. Configuration and
notebook cells were unchanged; the resulting models, scored observations and
training trace are byte-for-byte identical to version 13. The Kaggle forecasting
improvements above therefore also apply to version 14.

The completed workflow is source dataset **19** → builder **18** → processed
dataset **9** → training **14**. The earlier direct-upload rejection is retained
as historical evidence; the publication blocker is resolved. Default local
metadata uses the named dataset, and the cloud notebook no longer attaches
builder output directly.

Version 14 artifacts are in `artifacts_real_20260926/kaggle_v14/artifacts/`.
Verification and submission records are in
`artifacts_real_20260926/release_evidence/processed_user_refresh/`.

### Local evidence

- `artifacts_real_20260926/selection_standard/metrics.json`: validation-only run.
- `artifacts_real_20260926/selection_decision.json` and `frozen_config.json`:
  configuration and validation winners fixed before the test.
- `artifacts_real_20260926/final_test/metrics.json`: final metrics and protocol.
- `artifacts_real_20260926/final_test/paired_test_comparison.json`: paired errors
  and uncertainty estimates.
- `artifacts_real_20260926/final_test/training_trace.json`: exact observation IDs
  used for fitting, selection, residual scales and calibration.
- `artifacts_real_20260926/parquet_portability.json`: pandas 2/pandas 3 check.
- `artifacts_real_20260926/release_evidence/`: actual Kaggle publication evidence.

The final local model source SHA-256 is
`7a438153e20ad66422d7ab6bc93c4d3af420f5d46dfcf39332092371f67d5c3e`.
Local runtime: Python 3.14, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1.
Runtime versions and dataset/source hashes are stored with each run. A later
timestamp-dtype compatibility fix affects evidence comparison only; it changes
neither observations nor model predictions.

See [the runbook](KAGGLE_GUIDE.md) for source packaging, the identical local/online
builder cells and offline Kaggle training. Regenerate notebooks after changing
source so the expected fingerprints match the attached processed release.
