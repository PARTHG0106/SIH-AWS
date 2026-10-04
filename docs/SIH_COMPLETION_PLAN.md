# SIH26073 implementation and evidence status — 1 October 2026

The user authorized public comparison research and a separate, labelled synthetic
scenario pipeline. Real originals remain unchanged. The goal is complete,
demonstrable requirement coverage and measured comparisons, not an unsupported
claim to beat every public/private submission.

## Implemented and evaluated

- Public repository research: five SIH projects and four established libraries
  inspected at pinned commits. The package carries citation/hash metadata;
  third-party source bodies remain outside the release. Method-baseline results
  are distinguished from complete repository reproductions.
- Operational backend: incremental native-minute scoring with bounded per-station
  state, explicit cadence/availability monitoring, actionable explanations,
  unavailable health states, and a session API.
- Frontend: live historical replay and a one-click software-scenario stream,
  alert/health evidence, UTC plots, and separate real/synthetic evidence. The
  strict build and browser real/synthetic replay checks passed on 1 October.
- Scenario pipeline: copy genuine native windows into separate scenario arrays,
  preserve baseline IDs/hashes/values, record exact interventions, and avoid
  event/context leakage by partitioning source windows before generation.
- Evaluation: model selection and separate calibration completed, followed by
  one frozen February evaluation on temporal and station partitions. Point,
  event, classification and reliability metrics are preserved with method
  baselines in `artifacts_sih_final_20260930`.
- Packaging: documented final-directory selection, explicit inventory and
  dependency checks, use cases and deployment limits. An extracted verification
  build passed launcher and live HTTP checks; the final archive carries its own
  SHA-256 receipt and verification report.

The learned model's temporal point F1 is **0.688906**, compared with **0.455597**
for the frozen minute detector. Held-out Goodwin Creek F1 is **0.374488**, below
the frozen detector's **0.387186**, with **49.6451%** learned proposals on untouched
background rows. These results demonstrate a substantial station-transfer
limitation, not complete operational readiness. Detailed results and current
verification status are in [SIH_FINAL_RESULTS_20261001.md](SIH_FINAL_RESULTS_20261001.md).

## Protocol fixed before final evaluation

Only measured SURFRAD T/RH/station-pressure originals are eligible baselines.
Provider QC is not a hardware label. The synthetic background class is
`no_injection`, never `hardware_normal`. Scenario labels identify software
interventions, not proven physical causes.

| Partition | Sources | Purpose |
|---|---|---|
| Fit | Bondville/Fort Peck, 2023 | Fit scenario models and baseline statistics |
| Selection | Bondville/Fort Peck, January–August 2024 | Choose model/settings on synthetic validation only |
| Calibration | Bondville/Fort Peck, September–October 2024 | Confidence calibration and decision threshold |
| Final temporal test | Bondville/Fort Peck, February 2025 | Completed: 24 source windows; 129,600 scored scenario rows |
| Final station test | Goodwin Creek, February 2025 | Completed: 12 source windows; 64,800 scored scenario rows; excluded from scenario fitting/selection/calibration |

The frozen real forecaster was previously fitted/selected/calibrated on 2023–2024
observations. It is a separately declared pre-existing baseline, not a model
trained on scenario targets. January 2025 has already been examined and is not
reused as an untouched test. February was evaluated on 30 September 2026 at
12:43:57 UTC and is now consumed. Final February results must not be used for
further selection; subsequent improvements require a newly declared test period.

Source windows are disjoint, contain their own past context and are assigned
before scenarios. All variants of a source window stay in one partition. Test
definitions and difficulty distributions are fixed before evaluating February.
Failures and non-applied scenarios are reported, never silently discarded to
improve metrics. No point-adjusted F1 is used as a headline score.

The final comparison also reports a predeclared `combined_inspection_policy`:
the union of the frozen detector's candidates and the learned pattern model's
candidates. These remain separately labelled in the UI. The union's score is a
threshold ratio for ranking only, never a calibrated probability. This policy
is fixed before February evaluation; it is not selected on final results.

September 30 model selection compared multiclass boosting, separate binary/type
heads, ExtraTrees and Random Forest on development validation. The smaller
boosted classifier won the declared detection-PR-AUC/type-F1 objective. Final
confidence calibration separates monotone binary intervention probability from
conditional type temperature, preserving the detection ranking. Before sealing,
all 599,551 stored development feature rows were rederived exactly from the
hash-verified scenario copies, including the JSON-null runtime repair.

## Evidence and remaining work

The final report and its completed holdout receipt agree. The protocol records
211 nonoverlapping source windows across all five partitions; scenarios were
generated only after the source split, with 180 minutes of warm-up excluded from
scored targets. A read-only integrity audit on 1 October confirmed frozen source,
model, runtime, scenario-copy, manifest and receipt hashes. No evaluation was
rerun or model selected using February results.

Existing operational evidence includes unchanged-original stream/batch parity
and local runtime (`artifacts_live_verified_20260930/metrics.json`) and separate
v2 edge host measurements. These are software/runtime evidence, not real
hardware-fault accuracy. The 1 October full suite passed 398 tests (3 skipped,
6 subtests), and the strict build, browser real/synthetic replay, extracted
launcher and HTTP checks passed. Details are in the final-results report.

Remaining scientific/deployment work is station-transfer improvement under a new
evaluation protocol, verified live transport and field fault labels, and actual
hardware timing/energy measurements if an edge deployment is claimed. Physical
ESP32 energy, Indian AWS validation, confirmed degradation forecasts and
superiority over all other projects are not established by this release.
