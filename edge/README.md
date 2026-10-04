# Preliminary edge watchdog

`watchdog.py` is a small, dependency-free reference for an AWS device or gateway.
It reads **measured temperature, station pressure and independently measured
relative humidity**. It runs physical-domain/reporting-range, adjacent-step and repetition checks,
and an explicitly configured packet-cadence heartbeat. It is separate from the
hub's frozen forecasting detector and synthetic-trained scenario model.

The module uses portable Python syntax and `math`; it has no NumPy, pandas,
network, filesystem, sensor or radio dependencies. CPython tests pass. It has
**not been executed on MicroPython or ESP32 hardware**. Host timing and state size
below do not establish device timing, heap consumption, power or energy.

## Input and evidence contract

- One watchdog instance serves one `station|source` group. Pass actual packet time
  as nonnegative seconds from one consistent clock epoch, plus a dictionary with
  `temperature_c`, `pressure_hpa`, and `relative_humidity_pct`.
- Missing keys and `None` remain missing. Strings, booleans, NaN, infinity,
  non-increasing packet times and mixed station groups are rejected before state
  changes. Decode provider missing sentinels before calling the library and keep
  their original representation in the source log.
- Preserve the original packet, observation ID, timestamps, units, raw QC,
  source and hashes in the application log. The watchdog returns evidence without
  changing the input and retains no packet history.
- Physical-domain checks are temperature below absolute zero and negative RH.
  RH above 100% is a **reporting-range notice**, with possible supersaturation or
  measurement/reporting effects requiring source-specific review. Nonpositive
  station pressure is also a reporting-domain notice. These observations remain
  unchanged; the notices do not establish a hardware cause.
- Default step limits are **operator policies**, not calibrated performance
  thresholds: 5 °C, 6 hPa and 20 RH percentage points per adjacent sample. Exact
  repetition **strictly exceeding** 600 elapsed seconds triggers a review proposal
  (at 660 seconds for exact one-minute samples). Both checks use `value > threshold`,
  never `>=`. `flatline_seconds` accepts one number or a three-number tuple in
  temperature/pressure/RH order. All limits are configurable through the constructor;
  validate them separately for a real deployment. Do not substitute these rules
  for the hub's calibrated detector.
- A gap or a missing/range-noticed channel breaks its temporal context as an
  explicit conservative policy; a range notice is not proof of a physically
  impossible reading. This context treatment can differ from the hub.
  Step and repetition checks then report unavailable until adjacent usable
  context exists. There is no interpolation, no fault-free label and no lifetime
  or remaining-useful-life estimate.

An integration call uses values already obtained by your real sensor driver:

```python
from watchdog import PreliminaryWatchdog

watch = PreliminaryWatchdog(station_source, cadence_seconds=60,
                           step_limits=(5.0, 6.0, 20.0),
                           flatline_seconds=600, grace_seconds=0)
# timestamp_s and original_measurements are supplied by the acquisition driver.
evidence = watch.ingest(timestamp_s, original_measurements,
                        station_source=station_source)
```

The fixed state has three last values, three run anchors and three run-start
times, two timestamps, one saturating counter and bounded configuration. A group
identifier is at most 128 characters. No list grows with session duration.
Returned records are transient; accumulating them is the caller's separate
storage responsibility.

## Export an existing frozen reference policy

The optional host utility reads only an existing frozen `detector.json`. It copies
the per-group `abrupt_change` threshold unchanged and multiplies each
`flatline_minutes` threshold by 60. Durations start at zero on the first value;
the converted threshold retains strict `>` elapsed-time semantics. Exact native
one-minute cadence, zero timing grace and zero repetition epsilon are part of
this export. **No observations are read or thresholds fitted by the exporter.**

```powershell
.venv/Scripts/python.exe edge/export_policy.py --detector artifacts_minute_20260928/detector.json --groups "bon|noaa_surfrad" "fpk|noaa_surfrad" "gwn|noaa_surfrad" --out artifacts_edge_20260930/frozen_policy_v2.json
.venv/Scripts/python.exe edge/run_watchdog.py --cadence 60 --group "bon|noaa_surfrad" --policy artifacts_edge_20260930/frozen_policy_v2.json --input original_records.jsonl --output new_frozen_policy_results.jsonl
```

The policy includes the frozen artifact hash, calibration dates, original
threshold records, comparison operator, reference group and explicit channel
order. A requested group without its own reference uses the frozen
`pooled_seen_groups` reference only when exported explicitly; that fallback is
labelled and does not establish local calibration for the new station. The adapter
rejects a missing group or cadence override.

Every current result records one of `default_operator_policy`,
`provided_operator_policy`, or `frozen_signal_threshold_export`. The last means
**provided thresholds from the frozen hub reference**, not a separately calibrated
edge detector. Forecast, sustained-residual and learned scenario signals are not
exported; no full-model equivalence is claimed. The default ten-minute policy may
flag long natural repeats in rounded station readings. Neither a low nor a high
candidate count establishes real false-positive or hardware-fault accuracy.

## Host JSON-lines adapter

Use an existing JSON-lines file containing the original three measured fields
and either numeric `timestamp_s` or a timezone-aware ISO `timestamp`. A packet may
also include `station_id` and `source`, or `group`; these must match the declared
group. The adapter preserves each original record in its output next to the
watchdog evidence. It does not generate example observations or a live feed.

```powershell
.venv/Scripts/python.exe edge/run_watchdog.py --cadence 60 --group "bon|noaa_surfrad" --input original_records.jsonl --output new_watchdog_results.jsonl
```

Omit `--input`/`--output` to use stdin/stdout. Output files must be new paths.
`{"kind":"heartbeat","timestamp_s":...}` denotes a receiver-clock check, not
an observation. A `kind` of `observation` is the default. A heartbeat never inserts
rows or increments the received count. It reports the current overdue packet
count after the configured grace interval; repeated checks do not add that count
again. Before the first packet, availability is `waiting_for_first_packet` and
the missing count is unknown. A delayed but ordered original packet remains
admissible after a heartbeat.

## Device, sleep and hub responsibilities

1. Copy `watchdog.py` to the device and connect its call to an existing acquisition
   driver. Verify MicroPython's `math.isfinite`, clock resolution and serialization
   on that board. The source is a compatibility reference; no board port is
   claimed as tested.
2. Sample with the real sensor's documented settling/averaging requirements.
   Keep the declared cadence and original timestamps. Send explicit missing
   channels when an acquisition fails, and preserve raw driver/QC evidence.
3. Run the watchdog after acquisition, then save the original packet before
   sending it to the hub. Bounded durable queues, sequence IDs, acknowledgements
   and a radio transport belong to the acquisition application; they are not
   implemented by this policy module.
4. Use a board-supported mode that retains RAM, such as validated light sleep,
   if local temporal checks must continue. Deep sleep or reset discards this
   module's context: the next readings correctly start with unavailable step and
   repetition evidence. Automatic persistent-state restore is not implemented.
   Do not claim temporal continuity or fabricate missed samples after a restart.
5. Check packet silence on a receiver/gateway with its own clock while the device
   is asleep or disconnected. An offline device cannot establish its own absence
   from the hub. Configure heartbeat grace from verified transport behavior.
6. Send original observations to the hub's incremental API. Keep edge evidence
   in a separate field or log; never use it as a confirmed training label. Hub ML
   uses richer causal history and frozen models and is not numerically equivalent
   to this watchdog. Its current model was evaluated on NOAA native one-minute
   averages; a new AWS requires measured-channel, unit, cadence and deployment
   validation before making performance claims.

Sampling interval, radio batching and sleep state affect battery use. They must
be measured with the actual sensors, board, radio and power instrumentation.
There are no measured ESP32 power, energy, battery-life or duty-cycle savings in
this release.

## Reproducible host measurement

The adapter has a read-only mode for the already verified SURFRAD archive. It
selects exactly the requested station/day originals, verifies their acquisition
receipt hashes before and after the run, retains provider QC counts and maps only
the documented `-9999.9` missing sentinel to `None`. Every existing native record
is used. It neither fills missing minutes nor labels the hardware condition.

```powershell
.venv/Scripts/python.exe edge/run_watchdog.py --cadence 60 --measure-archive data/raw/surfrad_2025_january --day 2025-01-01 --stations bon fpk gwn --repetitions 5 --report artifacts_edge_20260930/host_measurement_default_v2.json
.venv/Scripts/python.exe edge/run_watchdog.py --cadence 60 --policy artifacts_edge_20260930/frozen_policy_v2.json --measure-archive data/raw/surfrad_2025_january --day 2025-01-01 --stations bon fpk gwn --repetitions 5 --report artifacts_edge_20260930/host_measurement_frozen_v2.json
.venv/Scripts/python.exe -m pytest tests/test_edge_watchdog.py -q
```

Use a new report path when reproducing; existing measurements are preserved.
The report records source and implementation hashes, source bytes, exact row
scope, environment, logical state bounds, CPython reachable state size and call
timing. Timing includes result construction but excludes parsing, serialization,
sensor sampling, transport and state-size inspection. The host load is not
controlled. Repeated timing passes each start a new instance and do not add new
distinct observations. Review counts are policy evidence, not accuracy or a
real false-positive rate.

Each run uses 4,320 unchanged original rows from three stations, repeated five
times for timing. The previously examined January records are used for runtime
and policy replay only; no threshold was chosen or tuned on them. The original
`host_measurement.json` and matching `source_default_v1/` copies retain the v1
default-policy reference, including its earlier `>=` repetition rule. Current v2
results use distinct paths and strict `>` semantics. Saved artifacts contain the
measurements and source lineage; they are authoritative for exact numbers.
Synthetic fixtures exist only in
`tests/test_edge_watchdog.py`.

Source interpretation follows the project's verified
[SURFRAD adapter](../src/awsad/data/surfrad.py) and
[real-data research policy](../docs/REAL_DATA_RESEARCH.md).
