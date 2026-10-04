# SIH26073 final scenario results and verification — 1 October 2026

The sealed learned model improves temporal detection over the existing minute
detector, but generalizes poorly to the held-out station. Its temporal point F1
is **0.688906**; Goodwin Creek F1 is **0.374488**, with candidates on **49.6451%**
of untouched-background rows. This supports a bounded synthetic-benchmark
improvement and exposes a substantial deployment limitation. It does not
establish field readiness, real hardware-fault accuracy or superiority over all
other submissions.

## Evaluation scope and freeze

The authoritative workspace run is `artifacts_sih_final_20260930`.
`final_test.json` records `final_test_complete` at
**2026-09-30T12:43:57.708662+00:00**. Its consumption receipt records exactly one
completed attempt. February 2025 is now consumed, as is the previously examined
January replay. Do not tune or select on February while describing it as an
untouched test; improvements require a newly declared independent test protocol.

Only original NOAA SURFRAD measured temperature, relative humidity and station
pressure were used as scenario baselines. Applied software changes have separate
targets; `no_injection` means no applied scenario, not verified fault-free
hardware. Missing readings, provider QC, source IDs and original hashes retain
their meanings. Scenario copies and repeated variants are not independent new
weather observations.

| Partition | Source stations and period | Source windows | Purpose |
|---|---|---:|---|
| Fit | Bondville / Fort Peck, 2023 | 96 | Model fitting |
| Selection | Bondville / Fort Peck, January–August 2024 | 63 | Model selection |
| Calibration | Bondville / Fort Peck, September–October 2024 | 16 | Probability calibration and threshold |
| Final temporal | Bondville / Fort Peck, February 2025 | 24 | Later-period evaluation |
| Final station | Goodwin Creek, February 2025 | 12 | Station excluded from scenario development |

The final split audit records **211 source windows and zero overlaps**. Windows
were assigned before scenario generation, including their context. The first
180 minutes of each window are excluded from scored targets. One incomplete
selection clock block was excluded and retained in the source audit. All variants
of a window stay in its partition. The frozen real-observation minute detector
is a separately declared pre-existing baseline.

The selected model is `multiclass_boosting`; decision threshold **0.48**,
conditional type temperature **1.119469420328474**, and binary logistic
calibration coefficients **[1.171029044539036, -1.3799707838243926]** were sealed
before test access. The freeze records exact float32 feature reverification,
including NaN, for **599,551** stored development rows. No model, calibration,
threshold or evaluation rule was changed after reading final results.

## Final learned-model results

Values below are rounded; `final_test.json` preserves full precision and the
per-pattern records. Point metrics use exact modified rows, without point
adjustment. Event recall requires a candidate in an applied modified interval;
it is not a count of confirmed physical faults.

| Metric | Temporal: Bondville / Fort Peck | Held-out station: Goodwin Creek |
|---|---:|---:|
| Scored scenario rows | 129,600 | 64,800 |
| Modified rows | 20,338 | 9,852 |
| Precision | 0.689892 | 0.241002 |
| Recall | 0.687924 | 0.839423 |
| Point F1 | 0.688906 | 0.374488 |
| PR-AUC | 0.783997 | 0.641430 |
| ROC-AUC | 0.896713 | 0.818640 |
| Applied events detected | 191 / 206 (92.7184%) | 96 / 105 (91.4286%) |
| Nine-pattern classification macro F1 | 0.566442 | 0.492323 |
| Ten-class accuracy, including `no_injection` | 0.889182 | 0.535463 |
| Untouched-background candidate rate | 5.9722% | 49.6451% |
| All unmodified-row candidate rate | 5.7559% | 47.3994% |
| Brier score | 0.072539 | 0.317532 |
| Expected calibration error (ECE) | 0.054217 | 0.349257 |

Untouched-background rate uses unchanged baseline variants. All-unmodified-row
rate also includes unmodified parts of altered scenarios, whose causal history
can contain an earlier intervention. These denominators differ. Neither rate
is a real hardware false-positive rate. Brier/ECE describe synthetic modification
probability, not the probability of a physical failure.

## Comparison on the same final scenario rows

These are the repository's named method implementations under the fixed
protocol, not reruns of complete competitor projects.

| Method | Temporal point F1 | Station point F1 | Temporal untouched-background candidates | Station untouched-background candidates |
|---|---:|---:|---:|---:|
| Rolling z-score | 0.163676 | 0.160256 | 0.8179% | 1.5278% |
| Step change | 0.233400 | 0.195856 | 11.6512% | 1.7593% |
| Isolation Forest | 0.234795 | 0.244103 | 0.9954% | 1.5741% |
| Frozen minute detector | 0.455597 | 0.387186 | 0.3858% | 0.2160% |
| Learned scenario model | 0.688906 | 0.374488 | 5.9722% | 49.6451% |
| Predeclared combined inspection policy | 0.637384 | 0.373065 | 6.1806% | 49.6914% |

The learned model gains **0.233309** absolute temporal F1 over the frozen
detector and loses **0.012698** station F1. The predeclared union detects
204/206 temporal events (99.0291%) and 102/105 station events (97.1429%), with
the background burden shown above. Its ranking score is a threshold ratio,
not a calibrated probability. High event recall alone does not justify it as a
deployment policy.

## Failures and limits

- Goodwin Creek's large background candidate rate and ECE **0.349257** show that
  calibration and specificity did not transfer adequately. The low station
  precision remains visible alongside its high recall.
- Temporal classification F1 is **0.243943** for bias, **0.209002** for scale
  error and **0.224299** for spikes. Held-out-station scale-error F1 is
  **0.091388**. Exact pattern typing remains unreliable for these cases.
- Temporal spike event recall is **16/24**. Among detected temporal events,
  median latency is **14 minutes** for stuck values and **10.5 minutes** for
  clipping. These conditional latencies do not include missed events.
- Thirteen clipping scenarios (10 temporal, 3 station) changed no values and
  are reported as not applied, outside applied-event denominators.
- One test month, three US stations and software interventions do not establish
  annual, Indian AWS, rare-weather or physical root-cause performance. Actual
  hardware status remains unknown. Source-window dependence also prevents
  treating all 194,400 scored scenario rows as independent examples.
- Neither a connected real AWS feed, validated degradation/lifetime forecast,
  ESP32 execution nor measured device power/energy is established here.

## Integrity and reproducible evidence

A read-only 1 October audit checked all **52** frozen source hashes and source-set
membership, selection/seal scripts, model and baseline bytes, frozen Python and
package versions, native/raw manifests, five scenario metadata/array pairs and
**2,110 scenario-copy hashes**. The completed receipt matches both the freeze and
final result, and `metrics.json` embeds the same final result. The audit did not
rerun the evaluation or modify sealed evidence.

- `artifacts_sih_final_20260930/final_test.json`: partition results, per-pattern
  details, reliability, provenance and split audit. SHA-256:
  `33853fcae4af867fc6c48825a4dd74e57635c347d961c70b473b830574008f81`.
- `artifacts_sih_final_20260930/frozen.json`: model, threshold, code and runtime
  seal. SHA-256:
  `57b71de4898dfd7ae2ac547aa8ae5a23d4b8e18e9a3e0d73239d65d12c32858f`.
- `artifacts_sih_test_registry/2e233b438d15c15f6d0c2b90e6aec4c9fc441c362328229273163eb806d0e7d3.json`:
  completed consumption record and final-result hash.
- `data/sih_scenarios_v2_reviewed_20260930/`: fit, selection, calibration,
  temporal-test and station-test scenario evidence referenced by the freeze.
- `docs/research/competitors_20260930/RESEARCH.md`: inspected implementation
  comparison and citations. Packaged `source_index.json` contains citation/hash
  metadata only; inspected third-party source bodies are not shipped.

Inspect existing results without consuming another test:

```powershell
$finalResult = Get-Content -LiteralPath artifacts_sih_final_20260930/final_test.json -Raw | ConvertFrom-Json
$finalResult.partitions.test_temporal.model.point
$finalResult.partitions.test_station.model.point
$finalResult.partitions.test_station.model.unmodified_baseline_candidate_rate
Get-FileHash -Algorithm SHA256 -LiteralPath artifacts_sih_final_20260930/final_test.json
.venv/Scripts/python.exe scripts/package_sih_release.py --check --pattern-dir artifacts_sih_final_20260930 --include-data --include-india --include-originals
```

## Operational evidence and continuation checks

`artifacts_live_verified_20260930/metrics.json` records stream/batch parity on
**720 unchanged original records** from two fixed October prefixes, with a
181-row detector-history bound per group. Mean one-row inference time was
**26.4727 ms**, p95 **44.3325 ms** and throughput **37.7748 rows/s**. The timing
includes frozen inference, state and JSON-safe conversion, excludes HTTP,
network and source loading, and was not measured under controlled host load.
This report establishes software compatibility and local timing, not fault
accuracy or a current browser run.

`artifacts_edge_20260930/host_measurement_default_v2.json` and
`host_measurement_frozen_v2.json` each use **4,320 distinct unchanged January
rows**, repeated five times for timing. Mean host ingest times were **8.7401 us**
and **8.8628 us**, respectively. They exclude sensor, transport and serialization
costs. The exported policy is in `frozen_policy_v2.json`; host timing/state are
separate from unmeasured ESP32 timing, RAM and energy.

The 1 October continuation completed these integration checks:

- Full suite: **398 passed, 3 skipped, 6 subtests passed** in 111.80 seconds.
  Focused packaging checks after the documentation/allowlist change: **31 passed,
  1 skipped** in 46.76 seconds.
- Final frontend build, including strict TypeScript, passed. Vite reports a
  roughly 790 kB JavaScript chunk; this is a bundle-size advisory.
- Browser synthetic spike stream at Bondville on 31 January, 00:00–06:00 UTC:
  **360/360 records**, **3 deliberate modified rows**, **7 frozen-detector
  candidates**, **3 learned-model candidates**. These are separate demonstration
  counts, not accuracy metrics.
- Browser unchanged replay, 00:00–03:00 UTC: **180/180 records**, no candidates,
  missing channels or unreported slots. The learned model correctly displays
  **Warming up**, requiring **181 contiguous minute records**. Absence of
  candidates does not establish fault-free hardware.
- Both benchmark partitions and the prominent station-transfer limitation were
  inspected in the browser. A screenshot is retained in the workspace at
  `out_sih_runtime_20261001/benchmark_station.jpg`.
- An extracted verification ZIP with **315 entries** passed the launcher's
  complete inventory and frozen-runtime checks. Its live HTTP smoke test passed
  **27 requests in 13.26 seconds**, covering local frontend assets, benchmark
  hashes/results, three US stations and six Indian demonstration stations.
- The HTTP check compared **360 original records** with the synthetic copy:
  baseline values, provenance and all **12 QC fields** matched; exactly three
  temperature rows were modified. Original replay packets contained no saved
  detector scores. It scored **360 synthetic and 181 real records**, verified
  warm-up/scored transitions with zero unavailable model statuses, and confirmed
  both temporary sessions were deleted. Counts are smoke evidence only.

The HTTP receipt and its one-off verifier are retained in the workspace at
`out_sih_runtime_20261001/release_http_smoke.json` and `verify_release_http.py`.
They are not part of the observation dataset or synthetic benchmark. Local
measurements include the tested host load and must not be presented as ESP32 or
controlled deployment timings.

The final ZIP is created after updating these notes. Its adjacent
`.zip.sha256` and `verification.json` record the exact archive and inventory
check, and compare its executable code, frontend, models and data with the
HTTP-tested build. Only these four evidence/deployment documents may differ
from that build. The sealed model and consumed February results remain unchanged.
