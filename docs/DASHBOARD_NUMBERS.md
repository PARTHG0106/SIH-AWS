# What the dashboard numbers mean

Verified against the saved January 2025 replay on 29 September 2026. The
dashboard reads original NOAA SURFRAD observations and separately saved detector
outputs. Its counts describe a historical replay. Candidate alerts remain
unconfirmed; hardware-fault status is unknown.

This guide uses `artifacts_minute_fresh_20260928`, station/source
`bon|noaa_surfrad` (Bondville), and the RH event on 31 January 2025. Selecting
another artifact, station or interval changes the numbers. All times below are
UTC.

## The current view: 09:00–13:46 on 31 January 2025

| Displayed number | Exact meaning |
|---|---|
| **287 observed rows** | 287 existing station records in this inclusive interval: 286 elapsed minutes plus both endpoints. Each record contains up to three measured values. |
| **167 candidate rows** | 167 distinct station minutes where at least one available detector signal exceeds its threshold. This window's 167 candidates all have the RH `flatline_minutes` reason. Multiple alerts on one row still count as one candidate row. |
| **0 missing readings** | None of the 861 channel values in these 287 rows is missing. Missing readings count individual T/RH/pressure fields; one row can contribute up to three. This is not a measurement-accuracy claim. |
| **0 unreported minutes** | No minute slots are absent between the existing records in this window. The card counts internal gaps only, not time outside the first and last returned record. An archive gap would not diagnose telemetry failure. |
| **287 rows have an anomaly score** | Each row has at least one available signal. This does not require every signal to be available; predictions need more history than a flatline counter. |
| **44,640 archived rows** in the sidebar | All Bondville rows in the loaded January artifact, before the window filter. Here this is 31 days × 24 hours × 60 records. It is one station's coverage, not a training-set size. |
| **60 minutes of context** | Display padding before and after the selected event. The event is 10:00–12:46, so the chart covers 09:00–13:46. This setting changes the view, not the detector or event boundaries. |

The chart measures temperature in **°C**, independently measured relative
humidity in **% RH**, and **station pressure in hPa**. Pressure is not reduced
to sea level. For this particular window, observed temperature spans 4.3–5.1 °C,
RH 98.2–98.3%, and pressure 976.8–979.1 hPa. Chart axes may be zoomed to the
observed range rather than starting at zero.

Green lines are observations. The separately labelled model lines are forecasts
for the displayed timestamp, built from earlier observations. Red circles mark
channel alerts at their unchanged observed values. Missing values and timestamp
gaps break lines; a line is never used to fill a missing observation.

## Why this RH event starts at 10:00

The saved measurements repeat **98.3% RH from 08:28 through 12:46**. There are
259 observations in that run, spanning 258 elapsed minutes. The flatline signal
counts elapsed minutes of exact repetition: the first observation has duration
zero. Missing values, changed values and non-minute gaps reset the counter.

The frozen Bondville RH threshold is **91 elapsed minutes**. A candidate requires
the signal to be **strictly greater** than its threshold; equality does not
trigger it. Earlier measurements are not backdated as alerts.

| UTC time | Measured RH | Exact-repeat duration | Duration / threshold | RH alert |
|---|---:|---:|---:|---|
| 08:28 | 98.3% | 0 min | 0 / 91 = 0 | No |
| 09:59 | 98.3% | 91 min | 91 / 91 = **1** | No |
| 10:00 | 98.3% | 92 min | 92 / 91 = **1.010989** | Yes |
| 12:46 | 98.3% | 258 min | 258 / 91 = **2.835165** | Yes |
| 12:47 | 98.2% | 0 min | 0 / 91 = 0 | No |

The last row's zero is the **flatline signal ratio**; its aggregate row score is
0.333333 because another available signal has a larger ratio. The selected event
contains 167 alerted observations from 10:00 through 12:46, spanning 166 elapsed
minutes. Its maximum channel score is **2.835164835164835**, reached at 12:46.
That means the repeat duration was about 2.84 times its empirical threshold.
It does not mean 283.5% confidence or a probability of failure.

At 10:00 the actual readings are **4.5 °C, 98.3% RH and 978.2 hPa**. The RH
one-minute forecast is also 98.3%, so its forecast error is zero. A successful
short forecast can coexist with a long exact-repeat alert: those checks measure
different properties. The temperature channel score is 0.071788, pressure is
0.276148 and RH is 1.010989; their maximum is the row's anomaly score.

The repeat began before the displayed 09:00 context. Its 08:28 start is supported
by the earlier archived rows; it is not inferred by extending the visible line.
Quantized readings and steady weather can produce repeated values. Independent
evidence is still needed to determine whether this event represents a fault.

## Signals, thresholds and scores

Let `y(t)` be the observation and `p1(t)` / `p60(t)` the forecasts for that time
using context ending one / 60 minutes earlier. All signal values are nonnegative.

| Saved signal | What it calculates | Units |
|---|---|---|
| `forecast_residual` | `abs(y(t) - p1(t))` | °C, hPa or RH percentage points |
| `abrupt_change` | Absolute change from the actual preceding minute; unavailable across a gap | °C, hPa or RH percentage points |
| `flatline_minutes` | Elapsed consecutive minutes of exactly equal measured values | Minutes |
| `hour_residual` | `abs(y(t) - p60(t))` | °C, hPa or RH percentage points |
| `sustained_deviation` | Absolute value of the mean **signed** 60-minute forecast error across the latest 30 consecutive observations | °C, hPa or RH percentage points |

The sustained signal is not the mean of absolute errors. Opposite signed
errors can cancel. It requires a complete 30-observation window within a
continuous minute sequence.

For a positive threshold, each signal's dimensionless ratio is
`signal / threshold`. A channel score is the maximum available ratio among
its five signals. `anomaly_score` is the maximum among the three channel
scores. Scores are **maxima, not averages**, and they are not probabilities.

An alert is decided from the unrounded raw comparison `signal > threshold`.
With positive thresholds, a ratio greater than 1 crosses the boundary; exactly
1 does not. Rounded displays can show 1.000 even when the saved value lies
slightly above it. The code also supports a zero threshold: a positive signal
maps to score 2, zero to score 0, and a missing signal remains missing. That
special case is not literal division; none of this artifact's thresholds is zero.

`channel__reason_codes` lists only signals that exceeded thresholds for that
channel. Row `reason_codes` also prefixes the channel name. An empty reason
means no available check crossed its threshold; it does not establish normal
hardware. `scored_available_signals` means some signal was available, not that
every model had sufficient past context. Missing scores remain unavailable.

The selected Bondville thresholds are:

| Signal | Temperature | Station pressure | Relative humidity |
|---|---:|---:|---:|
| Forecast residual | 0.5 °C | 0.3000000000000682 hPa | 4.86366148 percentage points |
| Abrupt change | 0.5 °C | 0.3000000000000682 hPa | 5.1 percentage points |
| Exact-repeat duration | 133 min | 84 min | **91 min** |
| 60-minute residual | 5.49094928 °C | 2.97354472 hPa | 33.75846354 percentage points |
| Sustained deviation | 5.01186974 °C | 2.47722151 hPa | 33.31370563 percentage points |

Values in the table are rounded where shown. The full-precision thresholds in
`detector.json` drive scoring. Thresholds depend on **station and source**, not
only the variable. Goodwin Creek was held out and uses `pooled_seen_groups`,
the thresholds calibrated on Bondville and Fort Peck, rather than thresholds
fitted to Goodwin Creek.

For example, the five RH checks at 10:00 are:

| Signal | Value | Frozen threshold | Ratio, approximately | Crosses? |
|---|---:|---:|---:|---|
| Forecast residual | 0 pp | 4.86366148 pp | 0 | No |
| Abrupt change | 0 pp | 5.1 pp | 0 | No |
| Exact-repeat duration | 92 min | 91 min | **1.010989** | **Yes** |
| 60-minute residual | 0.08680486 pp | 33.75846354 pp | 0.002571 | No |
| Sustained deviation | 0.05439514 pp | 33.31370563 pp | 0.001633 | No |

Here `pp` means RH percentage points. A change from 98.2% to 98.3% is 0.1
percentage point, not a 0.1% relative change.

## Counts for the whole January artifact

The top cards filter to the current view. The evidence JSON describes the whole
loaded artifact and must not be read as the window's counts.

| Metric | January value | Counting unit |
|---|---:|---|
| `native_observations` / `observed_station_minutes` | **133,920** | Existing records across the three station streams |
| `candidate_minutes` | **969** | Distinct station-minute rows with any channel alert |
| `candidate_fraction` | **0.007235663**, or **0.7235663%** | 969 / 133,920; candidate proportion, not a false-alarm rate |
| `channel_alerts` | **1,008** | Alerted observation/channel pairs; one station-minute can contribute up to three |
| `candidate_event_proposals` | **170** | Contiguous alerted intervals for a single channel, group and split |
| `review_proposals` | **173** | 170 event proposals plus three non-alert observations selected for review |
| `gaps_between_observed_records` | **0** | Number of gaps, not the number of missing minute slots used by the window card |
| `synthetic_observations` | **0** | No manufactured observations in this replay |
| `known_hardware_fault_labels` | **0** | No independently confirmed hardware labels; this does not mean zero faults |

| Station | Observed rows | Candidate rows | Candidate proportion | Event proposals |
|---|---:|---:|---:|---:|
| Bondville (`bon`) | 44,640 | 240 | 0.537634% | 57 |
| Fort Peck (`fpk`) | 44,640 | 543 | 1.216398% | 74 |
| Goodwin Creek (`gwn`) | 44,640 | 186 | 0.416667% | 39 |

Events break at gaps, non-alert or unavailable decisions, data-split boundaries
and monthly export boundaries. An event can contain several reasons over time;
its `max_score` is the largest **channel** score among its observations. An
event count is not a count of physical failures. Candidate rows on different
channels can overlap in time.

An event table's `start` and `end` are the timestamps of the first and last
alerted observations, inclusive. The review worksheet uses intervals ending
**exclusively** one cadence later: this example is `[10:00, 12:47)`. Neither
boundary is a confirmed hardware-fault onset or recovery time.

The development artifact `artifacts_minute_20260928` has **2,836,536 rows,
19,376 candidate rows and 4,100 event proposals**. It includes fitting/reference
periods and held-out-station replay, so its aggregate counts are not held-out
performance. Do not mix those totals with the January artifact.

## Provider QC and independent evaluation

For this verified SURFRAD format, raw QC **0** means the provider's checks
passed; values above zero flag a quality concern. Missing QC remains unknown.
QC acceptance is not a hardware-normal label, and rejection is not a hardware
failure diagnosis. All three channels in the selected event have QC 0.

Provider agreement counts partition each station/channel by provider status and
the saved channel alert: `accepted_alert`, `accepted_no_alert`,
`rejected_alert`, `rejected_no_alert`, and `unknown_or_unscored`. For January
Bondville RH, **224 accepted-alert + 44,416 accepted-no-alert = 44,640**. Both
rejected counts and unknown/unscored count are zero. These are agreement counts,
not true-positive/false-positive counts. Bondville's **240 candidate rows** are
the union over all channels, which differs from its **224 RH alerts**.

The later `independent_review_evaluation.json` reads **401,760 decision rows**:
three channel decisions for each of 133,920 native records. All 401,760 are
unreviewed, including 1,008 alerted channel decisions. It has **173 unknown
review intervals**, **zero reviewed observations**, **zero confirmed events**
and **zero reviewed background intervals**. Its status is
`unverified_no_reviewed_truth`; recall, false-alert metrics and detection delay
are `null` (unavailable), not zero performance.

The older `metrics.json` also embeds `real_event_evaluation`, created with an
empty decision frame when the review queue was generated. Its zero
`input_observation_rows` and `unknown_observation_rows` describe that empty
evaluation input, not the observation dataset. Use the separately saved later
evaluation for the full-input counts. Neither evaluation establishes fault
accuracy because independent reviewed truth is absent.

The updated dashboard verifies that separately saved report matches the loaded
detector/provenance and displays it in place of the empty-input placeholder.
If no completed report is attached, the placeholder is omitted and the missing
evaluation is stated explicitly. The underlying saved run files are unchanged.

## Model and calibration numbers in the evidence tab

| Setting or output | Meaning |
|---|---|
| Horizons **1 and 60** | Forecasts for the current timestamp using observations no later than one or 60 minutes earlier. |
| Context offsets **0, 5, 15, 60** | Offsets behind the forecast origin. The one-minute model needs values at `t−1, t−6, t−16, t−61`; the 60-minute model uses `t−60, t−65, t−75, t−120`. Inputs are past T/RH/station pressure and their differences. |
| Sustained window **30** | Thirty consecutive signed hourly forecast errors enter the sustained signal. |
| `reference_alert_budget = 0.005` | A nominal calibration tail budget of 0.5%, divided over 3 channels × 5 signals. It does not guarantee a replay alert rate or a false-alarm rate. |
| Quantile **0.9996666666666667** | `1 − 0.005 / 15`. Each threshold uses the empirical 99.9666667th percentile with NumPy's `higher` rule. |
| `reference_rows = 87,840` for Bondville | Actual eligible September–October 2024 reference signal values used for each Bondville threshold; 61 days × 1,440 minutes in this run. Other groups/signals have different counts. |
| `min_calibration_rows = 1,000` | Minimum required eligible reference values for a threshold, not the actual reference count. |
| `sample_per_month = 1,800` | Deterministic sampling cap for each station/month/horizon in model fitting or selection. Calibration uses its eligible reference values separately. |
| Fit counts **64,795–64,800** | Eligible sampled fitting examples for each channel/horizon; counts can differ because of channel QC. |
| Selection count **7,200** | Eligible sampled examples per channel/horizon used to compare gradient boosting with persistence. |
| `max_iter = 100` | Boosting iteration limit for training, not observed minutes. |
| `random_seed = 42` | Reproducibility setting, not a learned parameter or data count. |
| `max_cpu_threads = 4` | CPU concurrency limit. |

Fitting uses 2023-01-01 up to 2024-07-01 exclusive. Selection uses July–August
2024. Calibration uses September–October 2024. Bondville and Fort Peck supply
these phases. Goodwin Creek supplies none of them. The selected configuration
was frozen before January 2025 was scored; January has now been examined and
cannot be reused for further tuning while being called untouched.

All six channel/horizon models selected gradient boosting over persistence on
selection MAE. **MAE** is the mean absolute forecast error on the eligible
observations, in the channel's physical units. `model` is the selected model's
MAE and `persistence` predicts the unchanged earlier observation at `t−h`.
Lower MAE measures closer forecasts; it is not fault-detection accuracy.

For example, Bondville's January **60-minute temperature MAE** is **0.461978 °C**
for the model versus **0.614378 °C** for persistence, across **44,520** eligible
rows. The one-minute forecasts use 44,579 rows. The difference from 44,640 is
the missing pre-January model context at the start of this bounded replay:
120 rows for the 60-minute model and 61 for the one-minute model. The actual
observations still exist, and simpler signals can still be scored there.

## Original-record and provenance numbers

The 10:00 example is provider file `bon25031.dat`, original text **line 603**.
The event's final observation is **line 769**. `observation_id` combines the
raw-file SHA-256 and line number. The file hash is:

`0fb9eba956f2c0e1c619b5b432f0d4e9e6d86b5286073a53381cf4ab6c08c552`

Its retained provider URL is
[NOAA Bondville 31 January 2025](https://gml.noaa.gov/aftp/data/radiation/surfrad/Bondville_IL/2025/bon25031.dat).
This audit used the retained local record, not a new download. The acquisition
timestamp **2026-09-28 11:57:41 UTC** records retrieval; it is different from
the observation timestamp in January 2025.

`native_averaging_seconds = 60` means a one-minute sensor average, and
`timestamp_position = end_of_native_average` describes the provider's timestamp
convention. `native_file_version = 1` is a file-format version. These are not
forecast horizons or model scores. The retained Bondville header gives
**40.05° latitude, −88.37° longitude and 213 m elevation**; these metadata do
not enter the detector's forecasting features.

The January provenance manifest contains **93 daily original files**, **31 per
station**, totalling **32,680,789 bytes**, and three monthly native/scored shards.
File hashes and configuration/source/model fingerprints identify exact bytes;
they do not measure model quality. Raw admission and the saved artifact
verification check reconstruction separately from the dashboard's compatibility
checks. Observed fields, raw QC and lineage remain separate from predictions,
signals, candidate status and unknown fault labels.

## Sources for reproducing this explanation

- [Dashboard helpers](../app/real_dashboard.py) and [dashboard entry point](../app/streamlit_app.py)
- [Signal and score definitions](../src/awsad/minute_detection.py)
- [Event grouping and review intervals](../src/awsad/evaluation/real_events.py)
- [Replay metric definitions](../scripts/run_minute_detection.py)
- [Frozen thresholds and model selection](../artifacts_minute_fresh_20260928/detector.json)
- [Whole-run metrics](../artifacts_minute_fresh_20260928/metrics.json)
- [Event proposals](../artifacts_minute_fresh_20260928/candidate_events.csv)
- [Independent review evaluation](../artifacts_minute_fresh_20260928/independent_review_evaluation.json)
- [Provenance](../artifacts_minute_fresh_20260928/provenance.json) and [protocol](MINUTE_DETECTION_20260929.md)
