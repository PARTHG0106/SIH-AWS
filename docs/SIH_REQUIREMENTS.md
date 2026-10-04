# SIH26073 requirement-to-evidence map

Source: [official SIH 2026 statement](https://www.sih.gov.in/sih2026PS), entry
SIH26073. The workspace retains the inspected response in
`docs/research/competitors_20260930/source_receipts.json`; the submission package
includes citation and response-hash metadata in `source_index.json`, not the
third-party response bodies. Updated 1 October 2026.

The frozen February scenario evaluation is complete. The learned model improved
temporal point F1 to **0.688906**, but held-out Goodwin Creek F1 was **0.374488**
and its untouched-background candidate rate was **49.6451%**. This is a material
generalization failure. Executable coverage below does not establish field
readiness; see [final results and limitations](SIH_FINAL_RESULTS_20261001.md).

| Expected capability | Implementation | Evidence / remaining scope |
|---|---|---|
| Three meteorological inputs | Native measured T/RH/station pressure; derived model features only | SURFRAD provider verification, immutable originals and no missing-value replacement |
| Real-time anomaly alerts | `LiveMinuteDetector`, ordered packet API, incremental browser replay | Original-byte stream/batch parity and measured local latency; physical feed requires a transport adapter |
| Spikes/frozen values/fault patterns | Five frozen causal signals plus separate learned scenario-pattern model | Strict independently partitioned software scenarios; hardware causes remain uncertain |
| Communication errors | Explicit one-minute cadence and receiver-clock heartbeat | Missing fields and overdue slots separated; no fabricated input rows |
| Temporal/seasonal patterns | Training observations spanning 2023 and held-out later periods; short/long causal features | February 2025 temporal and station results reported separately; weak station transfer; annual or Indian generalization is not established |
| Multivariate consistency | Learned relationships among three channels and their past changes | Derived dew point is not independent consistency proof; sparse SURFRAD sites are not a local spatial network |
| Severity and confidence | Frozen evidence ratios/severity policy; separately calibrated scenario probability | Final ECE: 0.054217 temporal / 0.349257 held-out station; ratios are not probabilities; scenario confidence is not hardware-failure probability |
| Explainability | Channel, signal, threshold and magnitude, raw provenance, proposed inspection | Operator can trace each frozen alert to actual inputs; learned hints retain their separate source/uncertainty |
| Root-cause classification | Probable software-pattern typing | Similar observable patterns can share physical causes; no unsupported definitive diagnosis |
| Health and maintenance | Scored-row candidate activity, elapsed-time trend, coverage, service advisory | Requires declared minimum history; no-score is unavailable; no fabricated remaining life |
| Visualization | Live replay, one-click synthetic stream, real replay, synthetic India and benchmark pages | Strict build passed; browser real/synthetic streams and both benchmark partitions verified; backend warm-up status retained |
| Corrected values | Not automatically applied | Optional in the statement; source integrity takes priority |
| Scalability/deployability | Bounded per-group state, session limits, input validation, package inventory | Local prototype; durable transport, authentication and distributed store require deployment integration |
| Edge/energy | Separate portable preliminary watchdog and duty-cycle guidance | Host runtime/state measured; ESP32 execution and energy not measured |
| Executable code and use cases | CLI, API, scenario runner, packaging tool, tests, `SIH_USE_CASES.md` | 398 tests passed, 3 skipped; extracted archive launcher and live HTTP verification passed; workspace model is `artifacts_sih_final_20260930` |

## What comparison evidence supports

Five SIH projects and four established libraries were inspected at pinned
commits. Their useful methods and implementation limitations are documented in
the research report. Our benchmark compares implementations of transparent
method baselines on shared inputs and a fixed evaluation protocol. The learned
model exceeds the frozen minute detector's temporal F1 (0.688906 versus 0.455597)
but falls below its held-out-station F1 (0.374488 versus 0.387186). The comparison
does not rerun or rank complete competing repositories.

## Evidence locations

- `artifacts_live_verified_20260930/metrics.json`: original-source parity,
  input integrity, history bounds and local runtime.
- `artifacts_edge_20260930/host_measurement_default_v2.json` and
  `host_measurement_frozen_v2.json`: current watchdog host measurements, not
  physical device energy or fault accuracy; `frozen_policy_v2.json` records the
  exported threshold policy.
- `artifacts_sih_final_20260930/frozen.json`: completed model, code, dependency,
  split and calibration seal.
- `artifacts_sih_final_20260930/final_test.json` and `metrics.json`: completed
  February scenario results. The package stores this selected directory under
  the stable alias `artifacts_sih_20260930`; that alias is not the workspace run.
- `artifacts_sih_test_registry/2e233b438d15c15f6d0c2b90e6aec4c9fc441c362328229273163eb806d0e7d3.json`:
  one completed evaluation attempt and matching frozen/result hashes.
- `tests/`: software invariants and runtime/API contracts.

February 2025 is consumed, as is the previously examined January replay. Neither
may be reused for selection while called untouched. Further model improvements
require a newly declared independent test period and station protocol. Tests,
browser checks and extracted-package verification from this continuation are
recorded in the final-results report; they do not remove the scientific limits.
