# Native-minute detector and real replay — 29 September 2026

The approved continuation is implemented and has run locally: native-minute
detection, independently sourced event research, a review-gated benchmark, and
the active real-observation dashboard. **No real hardware-fault accuracy
improvement is claimed.** The evidence supplies no confirmed fault labels for
the evaluated period.

## Implementation

The native loader verifies the reviewed provider documents, original inventories,
acquisition receipts and raw-file hashes, then decodes all existing native minutes.
It supports explicit bounded acquisition periods. Missing data are never filled.

Six gradient-boosted change forecasters cover three channels × one/60 minutes,
selected against persistence on earlier validation observations. Predictors use
only past T/RH/station-pressure values and their differences. Station identity
and QC are not model predictors.

Five signals per channel cover forecast residuals, abrupt changes, unusually
long exact repeats, hourly residuals and sustained signed deviations. All windows
are causal. Gaps/missing readings interrupt continuity. Flatline duration and
QC eligibility carry across chunks; earlier readings are not backdated as alerts.

Thresholds use separate September–October reference observations, per
station|source, with a pooled fallback for Goodwin Creek. Contributing reference
context must have accepted QC, but acceptance is not a normal label. A 0.5%
nominal tail budget is divided over 15 channel/signal combinations. It is **not a
guaranteed false-alarm rate or failure probability**.

The active dashboard shows original measurements, separate predictions, reason
codes, gaps, source/QC inspection and review worksheets. The former simulated
feed is absent from this entry point.

## Frozen protocol

| Use | UTC interval | Stations |
|---|---|---|
| Fit | 2023-01-01 to 2024-07-01 exclusive | Bondville, Fort Peck |
| Select | 2024-07-01 to 2024-09-01 exclusive | Bondville, Fort Peck |
| Calibrate | 2024-09-01 to 2024-11-01 exclusive | Bondville, Fort Peck |
| Previously examined test | November–December 2024 | Excluded |
| Fresh frozen replay | 2025-01-01 to 2025-02-01 exclusive | All three |

Goodwin Creek is excluded from fitting, model selection and calibration. Its
earlier hourly results were already examined; the fresh evidence here is the
January period. Those raw files were acquired before training completed but
first parsed/scored after the detector was frozen. No settings were changed in
response to fresh results. January has now been evaluated and is not available
for further selection while retaining an untouched-test claim.

The original 2023–2024 archive has 3,099,744 actual records in 2,167 daily files.
The development view retains 2,836,536 before November 2024, including the
held-out station's replay. Deterministic monthly samples supply 64,795–64,800
eligible fitting examples per channel/horizon and 7,200 selection examples.
All six selected gradient boosting over persistence on selection MAE. That
comparison measures forecasting, not fault accuracy.

## Actual results

| Replay | Native station-minutes | Candidate station-minutes | Event proposals |
|---|---:|---:|---:|
| Development | 2,836,536 | 19,376 | 4,100 |
| Fresh January 2025 | 133,920 | 969 | 170 |

Development counts include fitting/reference periods and are not held-out
performance. An event proposal joins adjacent alerts for one channel/group.
It breaks at gaps, split boundaries and monthly output boundaries; it does not
represent a distinct physical failure.

| Fresh station | Observed minutes | Candidate minutes | Candidate fraction | Event proposals |
|---|---:|---:|---:|---:|
| Bondville | 44,640 | 240 | 0.54% | 57 |
| Fort Peck | 44,640 | 543 | 1.22% | 74 |
| Goodwin Creek | 44,640 | 186 | 0.42% | 39 |

954 of the 969 fresh candidate minutes are not minute00 records and would not
appear in the old hourly input selection. This is finer inspection coverage,
**not 954 additional true faults detected**. Most fresh exceedances involve
exact-repeat duration. Quantization and steady weather can also repeat values;
independent review remains essential. One season at three US sites does not
establish Indian AWS generalization.

The 170 fresh events plus three non-alert observation proposals remain unknown.
The evaluator reports `unverified_no_reviewed_truth`; fault recall, false alarms
and delay are unavailable. Provider-QC agreement counts are recorded separately.

## Evidence and benchmark

The [registry](research/surfrad_events_20260928/event_evidence_registry.json) and
[research report](research/surfrad_events_20260928/VERIFICATION.md) verify 23
downloaded responses, 24 receipts and 12 exact quotations. Five historical
incidents were documented; none supports labels for the current replay. Two old
Goodwin RH files retain flagged bad readings but use three-minute records and
a different sensor era. They remain research evidence, outside training and
benchmark scores.

`scripts/evaluate_real_events.py` accepts reviews with evidence references,
reviewer, UTC intervals and onset uncertainty. It joins shards and reports
event recall/delay and false alarms only inside reviewed intervals. `--signal`
compares frozen single-signal baselines on those same reviews without retuning.
Alert-only review samples cannot establish population accuracy.

## Artifacts and reproduction

- `artifacts_minute_20260928/`: trained models, configuration, source snapshot,
  66 scored shards, metrics, 4,100 event proposals and review worksheets.
- `artifacts_minute_fresh_20260928/`: identical frozen models/source, three fresh
  shards, 170 candidates, 173 review proposals and independent-review evaluation.
- Each has `artifact_verification.json` and `experiment_summary.json`.
  Model SHA-256: `744221e25d75e5eff28fbe463bde0186b080b62764139f80795c0b5058e33e2f`.
- All 56 original columns matched exactly for all **2,970,456** scored records.
  Nine source snapshot files, model bytes and all 69 scored shard hashes passed.
- `artifacts_minute_20260928_precalibration_fix/` preserves an interrupted
  preliminary attempt and is not the final run.

```powershell
python scripts/run_minute_detection.py --archive-dir data/raw/surfrad_original --cache-dir data/native_minute_20260928 --out artifacts_minute_new
python scripts/run_minute_detection.py --archive-dir data/raw/surfrad_2025_january --cache-dir data/native_minute_fresh_20260928 --out artifacts_minute_fresh_new --replay-only artifacts_minute_20260928 --start 2025-01-01T00:00:00Z --end 2025-02-01T00:00:00Z
python scripts/verify_minute_artifacts.py artifacts_minute_fresh_new data/native_minute_fresh_20260928
```

`kaggle/aws_minute_detection.ipynb` and `kaggle_minute/` are generated, validated
offline CPU entries for the existing processed-release workflow. They embed the
updated code and verify observation-release provenance separately. Bootstrap
passed on the actual portable local release (2,298 files, 2,167 indexed daily
originals), not a new remote execution. **Nothing was published or launched on
Kaggle in this continuation.** Existing hourly v14 is unchanged.

The final suite passed **228 tests and 6 subtests**, including both actual
artifact dashboards, future-data invariance, gap/chunk continuity, held-out
station independence, QC isolation, source integrity, notebook portability,
unknown labels and reviewed-event evaluation.
