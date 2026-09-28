# Real-observation audit and source research

**2026-09-26 follow-up:** live access now works. A new SURFRAD adapter and
real-observation forecasting path have been implemented; inspected primary
documents establish measured temperature, independently measured RH and station
pressure for the accepted scope. See [SURFRAD verification](research/surfrad_20260926/VERIFICATION.md).
The audit below remains the historical assessment of the legacy ISD/injected
pipeline. Its rejection of those artifacts still applies. Provider QC remains
separate from hardware-fault truth; unknown fault labels remain unknown.

Date: 2026-09-23. Project: SkyGuard AI / SIH26073.

**Decision: the existing processed dataset and demo are not eligible for a
claim of using only real observations and real fault labels.** Raw observations
exist locally, but the current pipeline interpolates values, injects faults,
derives a required input channel, and evaluates against constructed labels.

This report implements the user's revised data requirement as a documented
acceptance policy. It does **not** certify that the existing code, notebooks,
artifacts or datasets have been migrated to that policy. The audit is read-only;
it does not replace observations or delete the legacy files.

## Verification scope and limits

Completed in this session:

- Inspected every local NOAA station-year CSV header and file size.
- Counted actual records, missing sentinels, original quality codes and duplicate
  timestamps in three explicitly named 2024 files. This is a sample, not a
  statistical estimate for the full corpus.
- Hashed those sample files and the archived NOAA GHCNh documentation.
- Read the locally archived NOAA GHCNh v1.1.0 manual, dated March 10, 2026, its
  column dictionary, NAB's own data descriptions, and the project code paths.
- Traced source parsing, interpolation, synthetic labels, reanalysis references,
  model inputs, the demo generator and the dashboard.

Live verification was blocked. The sandbox denied socket access. Two browser
requests and two escalated read-only NOAA requests failed because their automatic
permission reviews timed out. This is an access failure, not evidence that a
provider is unavailable or its data are unsuitable. No replacement-source file
was downloaded in this session. Candidate sources below are not described as
newly verified. Local copies also have not been matched to a current provider
download in this session.

Machine-readable evidence: [real_data_audit.json](research/real_data_audit.json).
Reproduce the local inspection with `python scripts/audit_real_data.py`.
The script uses the standard library and does not need the ML environment.

## What is actually on disk

The local audit found **4,526 NOAA station-year CSVs**, **425 distinct station IDs
from filenames**, and **4,611,746,619 bytes** of station-year files. Years span
2013–2025, but this does not mean every station has every year or all variables.
All 4,526 headers include `TMP`, `DEW` and `SLP`; a column's existence does not
prove that it contains measurements. `MA1` appears in 2,754 headers. No
RH-prefixed column was found in this header inventory. This does not establish
what every physical instrument originally measured or what all raw report
strings may contain.

There are also 21 Open-Meteo raw files, 58 NAB data CSVs, **zero NCPOR files**,
and **zero files in the NASA SMAP/MSL directory**. NAB includes both real and
artificial collections; a corpus count must not be presented as a count of real
weather-station series.

Detailed raw-file examples, before any project preprocessing:

| 2024 station ID | Published rows | Duplicate timestamp rows | Present TMP | Present DEW | Present SLP |
|---|---:|---:|---:|---:|---:|
| 42182099999 | 2,888 | 28 | 2,887 | 2,887 | 2,756 |
| 42027099999 | 2,745 | 0 | 2,743 | 2,745 | 0 |
| 42809099999 | 19,635 | 2,761 | 19,631 | 19,630 | 2,763 |

“Present” means a parseable, non-sentinel number, including numbers carrying a
suspect quality code. It does not mean accurate, independent or fault-free.
For example, station 42182099999 has 27 temperature rows with QC code `2`
(suspect). These are 27 provider flags, not 27 proven sensor failures. Station
42027099999 has no non-sentinel SLP in this sample; it must not acquire pressure
observations through interpolation, a constant default or a reanalysis proxy.
Some files have an additional `MA1` field; decoding it requires the original
ISD field specification and a separate coverage audit. Its nonempty count is
not a count of valid station-pressure measurements.

## Findings that prevent a real-only claim

| Finding | Evidence | Required treatment |
|---|---|---|
| Manufactured fault values and event labels | `scripts/prepare_dataset.py`, `kaggle/build_data_builder.py`, `src/awsad/preprocessing/anomaly_injection.py` | Exclude injected rows and events from the submission dataset, model selection, headline evaluation and demonstrations. Rebuild from original observations; removing the label column does not undo a corrupted value. |
| Filled observations | `impute_hourly(..., max_gap=6)` interpolates short gaps | Keep a missing value and its provenance. Do not turn an absent report into a measurement. |
| Required RH is calculated | The ISD builder calls `dewpoint_to_rh(TMP, DEW)` | Store any calculated RH as a derived feature. It cannot establish an independent humidity-sensor input or humidity-sensor fault label. |
| Pressure meanings are conflated | `SLP` and Open-Meteo `pressure_msl` become `pressure_hpa` | Preserve sea-level pressure separately. Prefer source-documented station/barometric pressure for the physical sensor input. Never substitute one for the other silently. |
| Provider QC is promoted to fault truth | The parser/builder call suspect or erroneous QC codes “REAL ground-truth anomalies” and map dew-point QC onto RH | Preserve the original flag, source and semantics. It may support a provider-QC agreement task; it does not establish a hardware cause or a failure of another sensor. |
| Unflagged is treated as normal | Constructed binary labels use zero outside annotated/injected intervals | Keep unreviewed fault status unknown. Absence of a QC flag or maintenance record does not prove absence of faults. |
| Reanalysis is used as reference truth | Open-Meteo twins participate in spatial/reference paths | Exclude reanalysis from the primary observation and truth datasets. If ever used as model context, label it explicitly; it is not an independent sensor. |
| A “real” demo contains injected faults | `scripts/make_realdemo_csv.py` selects from the injected test split | Retire it from real-data demonstrations. Replay original, traceable observations instead. |
| Displayed “live” weather is generated | `app/streamlit_app.py`, `scripts/make_demo_csv.py` and timing scripts use random/sinusoidal readings | Use genuine timestamped observations for the hackathon demonstration. Call a historical replay a replay; do not call it a connected live station. |
| Completeness and certainty are overstated | Local processed metadata identifies a legacy snapshot; fault confidence uses rules/score mappings | No inherited accuracy, probability or station-coverage claims. Recompute measurements and report exactly what was tested. |
| NCPOR onboarding is not implemented | The raw folder is empty; the current preparation loader has NOAA and Open-Meteo branches only | Do not claim a working drop-in NCPOR ingestion path. Obtain a sample and implement its actual format. |

The prototype also masks QC-rejected values during preparation. A raw archive
must retain those original readings and flags, even when a particular training
view excludes them. Clipping, winsorisation or “repair” must not alter the
observation record used to investigate faults.

## What the primary-source documentation establishes

### NOAA GHCNh: a promising archive, not an automatic fix

The archived [GHCNh manual][ghcnh-manual] says that GHCNh replaces the legacy
Global Hourly / ISD product. Its data contain `temperature`,
`station_level_pressure`, `sea_level_pressure` and `relative_humidity`, each
with measurement, quality, report-type, source-code and original-station-ID
attributes. The 329-column v1.1.0 dictionary is stored locally in
`_verify_ghcnh/ghcnh_COLS.txt` and its accompanying PDF.

The manual defines station pressure as pressure observed at the actual
elevation, distinguishing it from sea-level pressure. Crucially, it states:

> “Depending on the source, relative humidity is either measured directly or
> calculated from air (dry bulb) temperature and dew point temperature.”

The RH measurement flag `D` means derived. Section VI explicitly identifies
sources 220, 221, 222, 223, 313, 314, 315, 322, 335, 343, 344, 346, 347 and 348
as deriving RH rather than measuring it directly. Missing `D` is not sufficient
proof of direct measurement: the actual source code, source documentation and
instrument history still matter. The source-code dictionary referenced by the
manual has not been retrieved in this session.

For six variables, GHCNh applies tests such as streak, spike, variance,
climatological and neighbour checks. These names describe quality-control
tests. They do not certify that the physical causes were latch-up, EMI, a
wiring swap or calibration drift. Some other fields retain legacy QC systems,
so the same numeric code must not be decoded without its source context.

The archived manual distinguishes, among others:

- `0`, `4`, `9`: gross-limit checks under the relevant ISD legacy scheme;
- `1`, `5`: passed all applicable QC checks under that scheme;
- `2`, `6`: suspect; `3`, `7`: erroneous;
- `U`, `P`, `I`, `M`, `R`: various edited, inserted, manually changed or
  computed values under the applicable legacy scheme.

Consequently, the current `GOOD_QC={0,1,4,5,9}` must not be described as a set
of independently verified, clean sensor observations. Source-level edited or
computed records require explicit provenance and exclusion from a strict
original-measurement view.

Preserve per-variable source IDs: integration can combine fields from different
input products into one station record. A merged station ID alone does not
establish that the three inputs came from one physical AWS instrument suite.

### NAB: useful provenance correction

NAB's own [data README][nab-data] identifies
`ambient_temperature_system_failure.csv` as ambient temperature **in an office**.
It describes `machine_temperature_system_failure.csv` as an industrial
machine's internal temperature, including a planned shutdown and machine
failure. These descriptions do not establish weather-station sensor failures.
`realAWSCloudwatch` refers to **Amazon Web Services**, and the corpus also has
explicit artificial-data directories. NAB cannot supply three-channel AWS
fault truth, and artificial subsets are ineligible under the user's policy.

## Source acquisition priorities

The following is a research queue, not a list of verified acquisitions.
The official entry points are given so every follow-up has a clear origin.
Do not write a production parser from the expected field names alone.

| Priority/source | Proposed role | Evidence required before acceptance | Current verification |
|---|---|---|---|
| [IMD Data Supply Portal][imd] | Indian target-domain observations and, if obtainable, maintenance/QC records | Sample raw files; actual T/RH/station-pressure fields; timestamps and cadence; station/instrument history; explicit access/usage terms; separately linked service/fault records | Access path described in existing project notes; no new retrieval or availability confirmation |
| [NCPOR/MoES repository][ncpor] | Official polar AWS observations as a separately reported domain test | Dataset-specific sample and metadata; independent RH measurement; pressure definition; licence; QC/maintenance records if offered | Existing notes report a CAPTCHA; zero local files; parser not implemented |
| [DWD Climate Data Center][dwd] | Candidate external observation corpus | Inspect official temperature/humidity and pressure documentation and station instrumentation; verify station-pressure fields and units; join only matching station/time records; preserve QC and gaps | Candidate; documentation, sample, coverage and terms not inspected this session |
| [NOAA GML SURFRAD][surfrad] | Candidate frequent meteorological observations with provider quality information | Verify actual T/RH/pressure fields, sensor documentation, cadence, flags and availability in downloaded files | Candidate; no source sample downloaded this session |
| [BSRN / PANGAEA][bsrn] | Candidate station observations and instrument metadata | Identify particular datasets with all three channels; verify pressure definition, timestamps, instruments and dataset-specific licence; obtain original records | Candidate; no DOI, station sample or three-channel coverage certified |
| [NOAA GHCNh][ghcnh] | Broad observation archive and potential improved pressure coverage | Source dictionary plus sample; per-variable measurement origin; RH derivation exclusions; pressure/QC provenance; station/AWS identification | Local official manual and column dictionary inspected; no GHCNh data acquired |
| Existing NOAA ISD | Preserve genuine temperature/dew-point/SLP observations and original reports | Reverify raw origin; decode additional pressure fields from the official specification; do not manufacture a missing measured RH input | Local headers and three record samples inspected; current three-input extraction is ineligible |

Global observations may support training, but foreign-network validation is not
Indian AWS validation. Keep network-specific holdouts and report the domain.
Do not assume that an airport, a SYNOP station, a radiation site or a polar
station is an IMD AWS installation just because it supplies weather data.

Open-Meteo/ERA5-style reanalysis, NASA POWER or other model products are not
substitutes for missing station measurements. Daily/gridded products and
spacecraft/server telemetry cannot fill gaps in a three-sensor AWS corpus.
USCRN, buoy and other network products remain candidates only if their actual
variable inventories satisfy the required channels; do not assume completeness.

## Binding acceptance policy for the revised dataset

1. **Never invent an observation.** Keep raw provider files unchanged. No
   synthetic faults, noise augmentation, sinusoidal weather, zero fills,
   forward/backward fills, interpolated measurements or reanalysis substitutions
   in the primary dataset or hackathon demo.
2. **Keep missing and unknown states explicit.** A missing archived reading is
   not a diagnosed telemetry failure. A missing label is not `normal`. Do not
   infer reporting schedules, station coordinates, elevation or sensor types
   when the source does not provide them.
3. **Require field-level provenance.** Every retained value needs provider,
   source product/version, station/original source-station ID, raw file SHA-256,
   original row/time, raw field, unit and raw measurement/QC attributes. Record
   retrieval time and URL when retrieval actually occurs. Do not backfill a
   fictional download timestamp for legacy files.
4. **Separate measured, calculated, edited and predicted values.** Mathematical
   features such as dew point may be calculated from available measurements and
   documented as derived features. They cannot replace a required absent input.
   Unit conversion is allowed with the provider's units and conversion recorded.
   Never call a model's proposed correction an observed value or ground truth.
5. **Preserve original time resolution.** Keep native reports, report types and
   duplicates in the raw view. If an hourly summary is necessary, expose it as
   a separate derived view with the aggregation rule, input row IDs, count and
   time span. Do not silently manufacture a complete hourly history.
6. **Use honest labels.** Store provider QC status separately from independently
   confirmed fault events. A nine-class fault label requires an actual record
   supporting that cause, station, channel and interval. Record uncertainty in
   event times; do not choose exact boundaries without evidence.
7. **Verify provenance and rights before accepting a source.** Archive the data
   dictionary, applicable terms, attribution and acquisition manifest. Do not
   bypass access controls or assume redistribution rights from public access.
8. **Treat unsupported records as ineligible, not repaired.** Publish measured
   coverage and exclusion counts. Unknown source semantics or missing required
   channels prevent a record from entering the strict three-channel view.
9. **Keep predictions out of truth.** Statistical models estimate relationships;
   their scores, confidence mappings and suspected causes are model outputs.
   An “assumption-free model” is not a claim this dataset policy can support.
10. **Rebuild and retrain after migration.** Existing artifacts, thresholds and
    results were developed with incompatible data. Do not carry their accuracy
    or calibration claims into the revised product.

Software unit-test fixtures are not observations and must stay confined to
tests. They must never be packaged as training/evaluation data, counted in
accuracy claims, or used to demonstrate a purported real station feed.

## Obtaining defensible real fault labels

Real weather observations and real fault labels are different acquisition tasks.
The inspected material does not establish a timestamped, nine-class, independent
AWS hardware-fault dataset. This is not a claim that none exists elsewhere.

The strongest next evidence would be provider maintenance/service tickets,
calibration records, sensor replacements, logger/communications incident logs,
or redundant-sensor comparisons linked to station, channel and time. A service
date alone does not establish the start of a drift; an outage log does not
automatically identify which sensor failed. Have an authorised domain reviewer
confirm the interpretation and preserve that evidence.

Until that evidence is available:

- Train appropriate unsupervised or self-supervised detectors on genuine
  observations with explicit quality/missingness controls; do not claim that
  an unlabelled training set is proven fault-free.
- Report provider-QC agreement only as provider-QC agreement, with the exact
  positive/negative/unknown definitions and without using QC fields as model
  predictors for that evaluation.
- Reserve sensor-fault precision/recall, cause accuracy, correction accuracy and
  maintenance prediction claims for independently annotated evidence.
- Exclude unknown event labels from supervised scores. Keep chronological,
  station and, where possible, source-network holdouts; tune no thresholds on
  the test set.
- Use a replay of traceable observed records to demonstrate scoring even if
  it contains no confirmed failures. Do not add a fault to make the demo look
  convincing.

## Concrete next work after access is restored

1. Verify the official IMD/NCPOR access paths and identify any supplied fault or
   maintenance records. Do not send requests or accept terms without the
   appropriate user authorisation.
2. Retrieve documentation and a small original sample from DWD and SURFRAD;
   verify all three fields and their measurement origin before selecting either.
3. Retrieve the GHCNh source dictionary and candidate station-year samples;
   measure retained coverage after excluding derived/edited required inputs.
4. Choose the corpus from those measured results. Build a provider-specific,
   native-resolution ingest path with immutable raw files and per-field lineage.
5. Replace both dataset builders, rebuild the processed data, replace simulated
   demos with real replay, and adapt model training/evaluation to the labels
   that actually exist. The current synthetic pipeline is not a shortcut.
6. Publish a release audit with accepted/excluded counts, terms, manifests,
   sample lineage, independently supported labels and newly computed metrics.

No replacement source, label collection, training run or deployment has been
certified by this audit. Live access and source-specific verification remain
required before those steps can honestly be called complete.

[ghcnh-manual]: https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf
[ghcnh]: https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
[nab-data]: https://github.com/numenta/NAB/blob/master/data/README.md
[imd]: https://dsp.imdpune.gov.in/
[ncpor]: https://data.ncpor.res.in/newhtml/
[dwd]: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/
[surfrad]: https://gml.noaa.gov/grad/surfrad/
[bsrn]: https://bsrn.awi.de/data/
