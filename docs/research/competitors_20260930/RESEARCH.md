# SIH26073 public implementation research — 2026-09-30

The official challenge explicitly permits simulated anomalies and says its
evaluation is on anomaly-injected data. A separate scenario benchmark is therefore
directly relevant. It must remain separate from real-observation replay and from
claims about confirmed hardware failures. The inspected public projects provide
useful ideas, but their reported scores do not establish a ranking against this
project: datasets, channels, splits, thresholds and metric definitions differ.

## Scope and reproducibility

Read-only public HTTP inspection was performed on 2026-09-30. No downloaded
implementation was imported, installed or executed, and no implementation code
was copied into the application. Source receipts retain request/final URLs,
retrieval timestamps, HTTP status, byte counts, SHA-256, encoding and full text.
All successful GitHub file inspections use immutable commit URLs.

- [Source receipts](source_receipts.json): original SIH page, exact extracted
  Description cell, GitHub searches, repository metadata/trees and inspected code.
- Submission packages include [source index](source_index.json), containing
  citation/hash metadata. Full inspected third-party source bodies stay in the
  research workspace and are not redistributed as project implementation.
- [Collector](collect_sources.py): public HTTP collection and numbered source display.
- [Offline audit](audit_receipts.py): reconstructs archived response bytes and
  verifies hashes; reproduces source-count and saved-metric arithmetic checks.
- Reproduce the offline audit from the project root:
  `.venv\Scripts\python.exe docs/research/competitors_20260930/audit_receipts.py`.
  It does not access the network or execute any downloaded source.

GitHub repository search returned **22** results for `SIH26073`, **77** for
`"automatic weather" anomaly`, and **78** for `"SkyGuard" weather`. These overlap;
they are neither a census of all solutions nor 177 independent competitors.
Five implementations were selected for detailed inspection because they expose
actual generators/evaluators and cover distinct designs. Four established
libraries were inspected for relevant methods. `anshrah-shaikh/AWS-Sentinel` was
inventoried but its notebook was not audited; no performance conclusion is made
about it or the remaining search results. An initial request to
`MET-Norway/titanlib` returned 404; the verified repository is `metno/titanlib`.

## Official problem statement and rubric

Source: [SIH 2026 problem statements](https://www.sih.gov.in/sih2026PS), entry
**SIH26073**, title **AI/ML-Based Intelligent Anomaly Detection for Automatic
Weather Stations (AWS)**; organization **Ministry of Earth Sciences (MoES)**,
department **India Meteorological Department**, category **Software**, theme
**Disaster Management**.

The complete original HTML response was retrieved at
`2026-09-30T11:34:04.255519+00:00`, SHA-256
`de0374415b7f27079de88b868905518d423550d569efcb46c2ba591042913ba7`.
Its exact Description HTML and extracted text are archived under
`official_problem_statement` in [source_receipts.json](source_receipts.json).
Extraction removes HTML tags/comments, decodes entities and normalizes whitespace;
it does not rewrite the source wording. The site's `Â°C` encoding artifact is
preserved there. The unit is rendered as °C in the summary below.

Three decisive source passages, verbatim:

> Develop an AI/ML-based intelligent anomaly detection system capable of automatically identifying abnormal, inconsistent, or faulty observations from Automatic Weather Stations in real time using only the following parameters:

> Expected Inputs Participants may use historical AWS datasets, simulated anomalies, or streaming sensor data containing the following meteorological parameters:

> Evaluation Criteria (To be evaluated in anomaly injected data)

| Criterion | Official weight |
|---|---:|
| Innovation & Novelty | 25% |
| Detection Accuracy | 20% |
| Real-Time Capability | 15% |
| Explainability | 10% |
| Scalability | 10% |
| Practical Deployability | 10% |
| Visualization/UI | 5% |
| Energy Efficiency | 5% |

Inputs are temperature, atmospheric pressure and relative humidity. Objectives
include spikes, frozen values, communication errors, temporal/seasonal learning,
multivariate consistency, explainable confidence, and possible sensor degradation
and maintenance needs. Expected outputs include alerts, severity/confidence,
root-cause classification, dashboard and sensor health. Corrected estimates are
optional. SHAP/LIME is preferable; ESP32 edge AI is suggested. The specified
deliverable is:

> Fully executable code with example usage and a document explaining various use cases

The challenge's request for a root-cause output does not turn an injected pattern
into verified hardware causation. This project's accepted distinction remains:
software scenario classification on the benchmark; uncertain diagnostic proposals
on real replay. A calculated dew point from T/RH is a useful representation, not an
independent sensor or independent thermodynamic validation of the same inputs.

## Inspected SIH implementations

### Aditya-Murugan1/SIH26073

Commit: [`268ef29cca1c596045670454a1b150071777c145`](https://github.com/Aditya-Murugan1/SIH26073/tree/268ef29cca1c596045670454a1b150071777c145).
No repository license was declared by the API or found in the inspected tree.

This is a substantial comparison target. Its
[benchmark builder](https://github.com/Aditya-Murugan1/SIH26073/blob/268ef29cca1c596045670454a1b150071777c145/scripts/build_benchmark.py#L40-L111)
partitions time and holds stations out before injection, keeps warm-up rows out
of scoring, writes immutable final benchmark hashes, and verifies regeneration.
Its [injector](https://github.com/Aditya-Murugan1/SIH26073/blob/268ef29cca1c596045670454a1b150071777c145/injection/injector.py#L25-L44)
varies magnitudes/durations across train/validation/test and includes positive and
negative spikes, flatline, drift, step, missing blocks, sentinel corruption,
multivariate/spatial inconsistency and composite events. Original baseline values
remain in `clean_<var>`, and regional software events are separate from injected
fault classes. These are valuable design ideas, independently implementable.

Limits in the inspected paths:

- [Data construction](https://github.com/Aditya-Murugan1/SIH26073/blob/268ef29cca1c596045670454a1b150071777c145/scripts/build_dataset.py#L33-L68)
  maps `alti * 33.8639` to `pressure_hpa`, maps `relh` to RH, and interpolates short
  gaps with an explicit `interpolated` quality flag. The
  [IEM provider form](https://mesonet.agron.iastate.edu/request/download.phtml)
  defines `alti` as “Pressure altimeter in inches”; a unit conversion does not
  establish station pressure. These inspected sources do not establish independent
  RH measurement. Preserve this project's verified SURFRAD input semantics and
  missing values rather than importing that data path.
- [Training](https://github.com/Aditya-Murugan1/SIH26073/blob/268ef29cca1c596045670454a1b150071777c145/scripts/train.py#L72-L101)
  uses validation for probability calibration, rule/threshold selection and
  feature-variant selection. It does separate final evaluation, but does not
  provide a distinct calibration partition in this path. Warm-up includes prior
  split history; this is causal for deployment but does not satisfy this project's
  stricter requirement that overlapping source histories remain in one partition.
- [Metrics](https://github.com/Aditya-Murugan1/SIH26073/blob/268ef29cca1c596045670454a1b150071777c145/ml/evaluation.py#L18-L54)
  report both point and any-overlap event recall. Its
  `false_alerts_per_station_day` is **false-positive rows / distinct station dates**,
  not deduplicated alert episodes or exposure-adjusted full days.
- [Edge evidence](https://github.com/Aditya-Murugan1/SIH26073/blob/268ef29cca1c596045670454a1b150071777c145/artifacts/edge/edge_metrics.json)
  honestly distinguishes a cross-compiled object and host runtime from hardware
  testing: `hardware_tested: false`, energy not measured, linked library helpers
  excluded from object size. This is a useful reporting standard, not proof of
  ESP32 latency or energy performance.

### Devansh-66/skyguard

Commit: [`584d15966337ea897cc564e6df378a1ed4d30f5a`](https://github.com/Devansh-66/skyguard/tree/584d15966337ea897cc564e6df378a1ed4d30f5a).
[MIT license](https://github.com/Devansh-66/skyguard/blob/584d15966337ea897cc564e6df378a1ed4d30f5a/LICENSE)
was inspected. The README's “Pre-Phase-0” status is stale relative to the code and
artifacts in this commit; the assessment here uses the actual files.

The [full evaluator](https://github.com/Devansh-66/skyguard/blob/584d15966337ea897cc564e6df378a1ed4d30f5a/evaluation/run_full.py#L76-L142)
combines per-channel learned residuals, conformalized scores, persistence,
neighbor differences and dedicated spike/noise channels. The
[metric implementation](https://github.com/Devansh-66/skyguard/blob/584d15966337ea897cc564e6df378a1ed4d30f5a/evaluation/metrics.py#L96-L179)
includes event recall, onset delay, amplitude-dependent probability of detection,
and quiet/active-weather alarm breakdowns. The
[edge report](https://github.com/Devansh-66/skyguard/blob/584d15966337ea897cc564e6df378a1ed4d30f5a/models/edge/report.json)
publishes cases where a CNN loses to simple rules as well as cases where it wins.
This motivates ablations and per-scenario reporting, rather than assuming a neural
model must improve every fault pattern.

Limits in the inspected paths:

- [Preparation](https://github.com/Devansh-66/skyguard/blob/584d15966337ea897cc564e6df378a1ed4d30f5a/scripts/prepare.py#L1-L10)
  explicitly reads an ERA5 export. It is a model/reanalysis background, not a
  verified measured AWS corpus. Other adapters exist in the tree but were not
  audited here; no conclusion about all its data paths is implied.
- [Metric thresholding](https://github.com/Devansh-66/skyguard/blob/584d15966337ea897cc564e6df378a1ed4d30f5a/evaluation/metrics.py#L200-L212)
  chooses the threshold using the evaluated frame's non-fault labels. This is an
  oracle budget operating point, not a frozen threshold deployed to unseen data.
- The [reference mask](https://github.com/Devansh-66/skyguard/blob/584d15966337ea897cc564e6df378a1ed4d30f5a/evaluation/run_full.py#L85-L107)
  is timestamp-only, while the same code explains that held-out stations can be
  `test` before the time cut. Thus reference and test rows can overlap for those
  stations in this evaluator. A strict station holdout needs separate admission.
- [Neighbor selection](https://github.com/Devansh-66/skyguard/blob/584d15966337ea897cc564e6df378a1ed4d30f5a/scripts/prepare.py#L117-L162)
  learns a graph from the entire provided frame unless the caller restricts it.
  Any data-learned graph must be fitted on the admitted training partition.
- `esp32_latency_ms_estimate` and `tensor_arena_est_bytes` in the edge report are
  estimates. They are not direct measurements of physical device energy.

### aditya0si/skyguard

Commit: [`66f0b9d861957d1e6c7a20e9f1b0e4d6d3e053e1`](https://github.com/aditya0si/skyguard/tree/66f0b9d861957d1e6c7a20e9f1b0e4d6d3e053e1).
[MIT license](https://github.com/aditya0si/skyguard/blob/66f0b9d861957d1e6c7a20e9f1b0e4d6d3e053e1/LICENSE)
was inspected.

It supplies explicit synthetic climate profiles, six scenario topologies,
Isolation Forest, spectral residual, an autoencoder and a weighted ensemble.
Its [ensemble](https://github.com/aditya0si/skyguard/blob/66f0b9d861957d1e6c7a20e9f1b0e4d6d3e053e1/src/skyguard/models/ensemble.py#L60-L95)
exposes component scores and physics reasons; this is useful operator-facing
structure. However the
[evaluation loop](https://github.com/aditya0si/skyguard/blob/66f0b9d861957d1e6c7a20e9f1b0e4d6d3e053e1/evals/run_eval.py#L40-L97)
concatenates generated profiles, calls `model.fit(combined_df)`, then scores that
same frame. It reports **seven channels**, including wind, solar, precipitation
and battery, so it is not a three-input held-out SIH benchmark.

The [metrics implementation](https://github.com/aditya0si/skyguard/blob/66f0b9d861957d1e6c7a20e9f1b0e4d6d3e053e1/src/skyguard/benchmark/metrics.py#L53-L84)
expands one detected point to every point in the true anomaly segment for PA-F1.
This metric should not be substituted for ordinary point F1 or detection delay.
The evaluator's `latency_ms` times fitting plus batch scoring together; it is not
per-request online inference latency. “Calibrated” in the ensemble name does not
by itself establish held-out probability calibration: the inspected implementation
normalizes configurable weights and clips scores to [0,1].

### D-Tharun/SIH26073-Team-Kestrel

Commit: [`26346f85721b90a034ae0148acf999e80dbbad89`](https://github.com/D-Tharun/SIH26073-Team-Kestrel/tree/26346f85721b90a034ae0148acf999e80dbbad89).
Its [LICENSE](https://github.com/D-Tharun/SIH26073-Team-Kestrel/blob/26346f85721b90a034ae0148acf999e80dbbad89/LICENSE)
is only the text “Apache License 2.0”, not the complete standard license text;
GitHub labels it `NOASSERTION`. Do not assume a fully documented reuse grant.

The [training path](https://github.com/D-Tharun/SIH26073-Team-Kestrel/blob/26346f85721b90a034ae0148acf999e80dbbad89/server/training/train.py#L38-L91)
reads Jena climate T/pressure/RH, drops missing rows, samples every sixth row,
constructs 22 features, then applies chronological 70/15/15 splits. Scaling is
fitted only on training data, and sequence windows are constructed separately
inside each split ([training code](https://github.com/D-Tharun/SIH26073-Team-Kestrel/blob/26346f85721b90a034ae0148acf999e80dbbad89/server/training/train.py#L345-L417)).
It combines Transformer-VAE, Anomaly Transformer and Isolation Forest with physics,
temporal, multivariate and spatial rules. This is concrete implementation beyond
a mock dashboard; its model complexity is not evidence of superior detection.

The [test injector](https://github.com/D-Tharun/SIH26073-Team-Kestrel/blob/26346f85721b90a034ae0148acf999e80dbbad89/server/training/evaluate.py#L80-L125)
corrupts randomly selected already-built windows and recomputes their derived
features. That evaluates independent corrupted windows, not one consistent stream
of events across overlapping windows. The spatial test uses hard-coded
[dummy buddies](https://github.com/D-Tharun/SIH26073-Team-Kestrel/blob/26346f85721b90a034ae0148acf999e80dbbad89/server/training/evaluate.py#L141-L175),
so that result does not validate a real spatial network. If saved test windows are
absent, its fallback explicitly warns they may overlap training. These are reasons
to test a frozen, contiguous stream with source/event admission before injection.

### Luciefer-555/SIH26073-Weather

Commit: [`8c56ccaaff67d45ffea4039567c78674313929e4`](https://github.com/Luciefer-555/SIH26073-Weather/tree/8c56ccaaff67d45ffea4039567c78674313929e4).
No license was declared by the API or found in the inspected tree.

Its [generator](https://github.com/Luciefer-555/SIH26073-Weather/blob/8c56ccaaff67d45ffea4039567c78674313929e4/generate_dataset.py#L34-L76)
creates 5 stations × 60 steps minus 5 omitted rows = **295 synthetic rows**,
with six meteorological channels. Its regional “REAL EVENT” is deliberately
generated, not an observed weather event. The
[default detector](https://github.com/Luciefer-555/SIH26073-Weather/blob/8c56ccaaff67d45ffea4039567c78674313929e4/ml-detector/detector.py#L109-L182)
is past-history rolling z-score plus exact-value persistence and drift checks;
Isolation Forest is an optional alternative.

Its useful engineering detail is the
[two-pass replay](https://github.com/Luciefer-555/SIH26073-Weather/blob/8c56ccaaff67d45ffea4039567c78674313929e4/run_on_dataset.py#L44-L71):
populate all stations for a timestep before spatial classification, avoiding
station-order dependence. The [classifier](https://github.com/Luciefer-555/SIH26073-Weather/blob/8c56ccaaff67d45ffea4039567c78674313929e4/ml-classifier/classifier.py#L112-L152)
returns `uncertain` when there are no neighbors. Confidence otherwise comes from
a hand-written agreement formula, not observed calibration.

Its [dropout checks](https://github.com/Luciefer-555/SIH26073-Weather/blob/8c56ccaaff67d45ffea4039567c78674313929e4/run_on_dataset.py#L98-L123)
expect omitted rows to cause no crash and no unexpected alert. They do **not**
demonstrate communication-failure detection. That requires a known reporting
schedule, elapsed-time logic and a documented lateness allowance, independently
of whether a new measurement arrives.

## Published numbers: context, not a league table

These are values read from pinned repository artifacts, **not model reruns by
this research**. Offline recomputation verifies only the stated arithmetic.

| Artifact | Read result | What it measures / limit |
|---|---|---|
| [Aditya-Murugan final metrics](https://github.com/Aditya-Murugan1/SIH26073/blob/268ef29cca1c596045670454a1b150071777c145/artifacts/final_evaluation/metrics.json) | 94,367 evaluated rows; point F1 0.586757; any-overlap event recall 0.900596 | Injected METAR-based benchmark; point F1 reproduced from TP=3,385, FP=741, FN=4,027. Event recall and point F1 answer different questions. |
| [aditya0si results](https://github.com/aditya0si/skyguard/blob/66f0b9d861957d1e6c7a20e9f1b0e4d6d3e053e1/evals/results.json) | Spectral residual F1 0.2283; PA-F1 0.7141 | Same-data fit/score on synthetic profiles; point adjustment multiplies the reported F1 by 3.1279. Not an improvement in causal detection. |
| [Kestrel results](https://github.com/D-Tharun/SIH26073-Team-Kestrel/blob/26346f85721b90a034ae0148acf999e80dbbad89/server/artifacts/evaluation_results.json) | Accuracy 82.84%; precision 22.43%; recall 99.05%; F1 36.58% | Saved ensemble evaluation on injected windows; high recall alone conceals alarm burden. |
| [Devansh edge report](https://github.com/Devansh-66/skyguard/blob/584d15966337ea897cc564e6df378a1ed4d30f5a/models/edge/report.json) | CNN stiction detection 0.0 vs rules 1.0; CNN step detection 0.97 vs rules 0.996 | Its own injected edge cases and threshold convention. Demonstrates why per-pattern losses should be reported, not relative efficacy on our dataset. |

No claim that SkyGuard is better than these repositories is supported by this
inspection. Such a claim requires admissible implementations run on the same
frozen data, input set, hardware, budgets and scoring rules. Unlicensed code was
not copied or run. Equivalent standard-method baselines implemented independently
should be named for their algorithms, not presented as reproductions of these
repositories.

## Established libraries and applicable ideas

| Project and pinned commit | Inspected evidence | Applicable idea and boundary |
|---|---|---|
| [metno/titanlib](https://github.com/metno/titanlib/tree/18a4fac903a0ae474a318acacf8af91057490f03), LGPL-3.0 | [README](https://github.com/metno/titanlib/blob/18a4fac903a0ae474a318acacf8af91057490f03/README.md), [buddy check](https://github.com/metno/titanlib/blob/18a4fac903a0ae474a318acacf8af91057490f03/src/buddy_check.cpp#L90-L115), [spatial consistency test](https://github.com/metno/titanlib/blob/18a4fac903a0ae474a318acacf8af91057490f03/src/sct.cpp#L23-L68) | C++ spatial QC with radius, minimum neighbor counts, elevation differences and validity checks. Use only when genuine contemporaneous neighbors and metadata exist. Sparse SURFRAD sites do not become a validated buddy network by adding a distance formula. QC flags do not establish hardware cause. |
| [HPI-Information-Systems/TimeEval](https://github.com/HPI-Information-Systems/TimeEval/tree/7c3622757b1a3afa9a84cbc97ce132d9e379f715), MIT | [resource limits](https://github.com/HPI-Information-Systems/TimeEval/blob/7c3622757b1a3afa9a84cbc97ce132d9e379f715/timeeval/resource_constraints.py#L35-L87), [experiment execution](https://github.com/HPI-Information-Systems/TimeEval/blob/7c3622757b1a3afa9a84cbc97ce132d9e379f715/timeeval/_core/experiments.py#L95-L161) | Persist parameters, raw scores, timing and errors; repeat experiments; separate train/execute timing. Its container resource limits apply to the Docker adapter, not every API. A benchmark framework does not automatically supply valid AWS labels or deployment calibration. |
| [TheDatumOrg/TSB-AD](https://github.com/TheDatumOrg/TSB-AD/tree/6beac72e11d1155ade40870492c00d0d1cfdcaaf), Apache-2.0 | [tuning runner](https://github.com/TheDatumOrg/TSB-AD/blob/6beac72e11d1155ade40870492c00d0d1cfdcaaf/benchmark_exp/HP_Tuning_M.py#L30-L72), [evaluation runner](https://github.com/TheDatumOrg/TSB-AD/blob/6beac72e11d1155ade40870492c00d0d1cfdcaaf/benchmark_exp/Run_Detector_M.py#L29-L87), [metrics](https://github.com/TheDatumOrg/TSB-AD/blob/6beac72e11d1155ade40870492c00d0d1cfdcaaf/TSB_AD/evaluation/metrics.py#L3-L38) | Separate tuning/evaluation file lists, deterministic seeds, saved scores and standard/range metrics. `pred=None` explicitly selects an oracle threshold for dependent metrics; pass frozen predictions for deployable F1 and label oracle summaries as such. Its general time-series corpus is not AWS hardware-fault truth. |
| [TUW-GEO/pytesmo](https://github.com/TUW-GEO/pytesmo/tree/054e2c4d2915d5e762af7a919a2b521476c21415), BSD-style terms in [LICENSE.txt](https://github.com/TUW-GEO/pytesmo/blob/054e2c4d2915d5e762af7a919a2b521476c21415/LICENSE.txt) | [temporal collocation](https://github.com/TUW-GEO/pytesmo/blob/054e2c4d2915d5e762af7a919a2b521476c21415/src/pytesmo/temporal_matching.py#L184-L231), [bootstrap intervals](https://github.com/TUW-GEO/pytesmo/blob/054e2c4d2915d5e762af7a919a2b521476c21415/src/pytesmo/metrics/confidence_intervals.py#L75-L164) | Return source timestamps and matching distances when aligning data. Default nearest matching can include future records and mean matching produces aggregates, so live causal observation admission needs stricter matching. Its bootstrap resamples points; use station/event/source-block resampling for correlated scenario data. This is a geospatial validation toolkit centered on soil moisture, not a competitor AWS fault model. |

## Concrete improvements justified by the inspection

1. **Freeze data admission before scenario generation.** Preserve verified
   measured T/RH/station pressure, raw flags, URLs and original hashes. Allocate
   source station/time windows and their complete overlapping histories to one
   partition. Use fit, model-selection, probability/threshold-calibration and
   final-evaluation partitions; changing the injection seed alone is not an
   independent background. January 2025 has already been examined here and must
   not be renamed an untouched selection holdout.
2. **Generate events in a continuous source window, then derive features.** Keep
   baseline and scenario values in separate columns, event IDs, start/end times,
   channel masks, magnitude, duration and seed. Include spike, step, drift,
   persistence, noise and missing-block interventions. Label an unmodified
   background `no_injected_scenario`, not confirmed normal hardware. Include
   coherent T/RH/pressure changes as software weather-like controls, with the
   same synthetic designation. Never promote such controls to real weather truth.
3. **Publish operational metrics at a frozen threshold.** Report ordinary
   point precision/recall/F1/PR-AUC, event recall, alarm-episode precision, onset
   delay including missed-event counts, channel attribution, and per-type /
   duration / amplitude results. Show alarm episodes per observed station-hour
   or day with cadence/gap rules documented. Keep flagged-row rates distinct from
   alert episodes. Add event/source-block confidence intervals. PA-F1 and
   test-label-selected thresholds cannot be headline deployment results.
4. **Compare simple and learned methods on identical admitted data.** Include
   median/MAD or Hampel, persistence/rate rules, Isolation Forest, and the proposed
   learned classifier/ensemble. Ablate temporal, multivariate and physics-derived
   features. Benchmark every variant at the same alarm budget and report failures
   on subtle drift and plausible extremes. Do not label these independently
   implemented baselines as replicas of inspected repositories.
5. **Calibrate claims as well as scores.** Separate evidence/rule confidence,
   empirical synthetic-class probabilities, and unknown hardware cause. Report
   Brier score and reliability bins only on the admitted benchmark target and
   with bin sample counts. A float between 0 and 1, SHAP attribution or a
   neighboring agreement formula is not automatically a fault probability.
6. **Measure the actual runtime path.** Separate fit time, feature computation,
   batch throughput and online p50/p95/p99 latency. Include state/memory/model
   bytes and warm-up behavior. State host, thread count, observation count and
   repetitions. If an ESP32 model is exported, distinguish model/object size,
   linked firmware size, host timing, emulator estimates and physical readings.
   Energy remains unmeasured until measured on specified hardware.
7. **Make explanations and health actionable without inventing certainty.**
   Show the original trace, current score/threshold, affected channel, reason,
   scenario-vs-replay provenance, missing/unknown states and review disposition.
   Aggregate repeated proposals into events. Maintenance output can describe
   worsening evidence and recommended inspection; remaining useful life and
   confirmed causes require their own verified targets.

These are implementation and evaluation priorities derived from inspected
evidence. Completion and efficacy must be demonstrated by this repository's own
artifacts; this research report does not certify that those improvements have
already been implemented.
