# Dataset audit and training-data policy (SIH26073)

**Audit date:** 2026-09-20  
**Scope:** sources proposed for AI/ML anomaly detection in Automatic Weather
Stations (AWS), with the detector contract restricted to `temperature_c`,
`pressure_hpa`, and `relative_humidity_pct` plus physics-derived features.

This document is the data decision record for the project. A source is not
accepted merely because it is downloadable: we check whether it is a real
station observation, whether its timestamps and variables match the problem,
whether redistribution is allowed, and whether it can provide defensible
evaluation labels.

## Decision summary

| Source | Decision | Role in this project | Reason / limitation |
|---|---|---|---|
| [NOAA ISD Global Hourly](https://www.ncei.noaa.gov/data/global-hourly/access/) | **Use** | Primary observation corpus / AWS-compatible proxy | Real surface-station reports and a reproducible global archive. Indian station metadata does not prove that every record is an IMD AWS installation, so headline claims call this a surface-station proxy. ISD `DEW` is used to derive RH; `SLP` is sea-level pressure, not guaranteed station pressure. Duplicate report types must be field-wise aggregated. |
| [IMD Data Supply Portal](https://dsp.imdpune.gov.in/) | **Use when authorised** | Highest-relevance external holdout | Official AWS data, but enrolment/payment/access terms prevent a public reproducible bundle. Never redistribute files without permission. |
| [NCPOR/MoES repository](https://data.ncpor.res.in/newhtml/) | **Use as separate polar test** | External domain test | Contains relevant Maitri/Bharati AWS records. CAPTCHA/manual access and a different polar regime mean it should not be silently mixed into the land-network headline score. |
| [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api) | **Keep separate** | Reanalysis/reference and optional augmentation | ERA5-style model/reanalysis values are smooth and are not independent AWS sensor truth. Same-location “twins” can make injected faults unrealistically easy. They are tagged `source_role=reference_reanalysis`. |
| [NOAA SURFRAD](https://www.gml.noaa.gov/ozwv/surfrad/) | **Optional** | High-frequency fault-stress/domain test | Real surface observations, but channel mix and station network differ; RH/pressure coverage must be checked per file. |
| [NOAA NDBC](https://www.ndbc.noaa.gov/) | **Optional** | Marine-domain robustness test | Real buoy meteorology; RH may need derivation from dew point and marine dynamics differ from land AWS. |
| [NOAA USCRN](https://www.ncei.noaa.gov/products/land-based-station/us-climate-reference-network) | **Optional** | Channel-specific pretraining/test | High-quality 5-minute temperature/RH, often no pressure; do not use as a complete three-channel AWS corpus. |
| [DWD CDC](https://www.dwd.de/EN/ourservices/cdcftp/cdcftp.html) | **Optional** | Cross-network test | Open hourly station observations with useful T/RH/pressure coverage; licensing and schema normalization must be recorded before bundling. |
| [Environment Canada](https://climate.weather.gc.ca/) | **Optional** | Cross-network test | Real government observations; station pressure/altitude conventions require normalization. |
| [NOAA GSOD](https://www.ncei.noaa.gov/products/land-based-station/global-summary-of-day) | **Reference only** | Long-term drift/context checks | Daily summaries are too coarse for the hourly fault detector. |
| [Numenta NAB](https://github.com/numenta/NAB) | **Benchmark only** | Generic streaming-method sanity check | Labeled but mostly non-weather/univariate. `realAWSCloudwatch` means Amazon Web Services telemetry, not Automatic Weather Stations. Never report NAB as SIH weather accuracy. |
| NASA SMAP/MSL / SWaT / WADI / UCR-TSB | **Transfer benchmark only** | Optional representation/streaming comparison | Not weather stations and not suitable as primary AWS training data. |
| `imdlib` | **Do not use as sensor truth** | Optional gridded climate context | IMD gridded products are not station telemetry and cannot supply AWS fault labels. |

## Primary corpus and provenance

The online Kaggle builder downloads NOAA station-year files directly inside
Kaggle, parses them with the checked-in parser, and writes a new processed
bundle. The source package version is asserted before download so a stale
attached dataset cannot silently reintroduce the old duplicate-timestamp bug.
The builder records `source_role`, `rh_source`, `pressure_source`, parser
version, station catalogue, split boundaries, and SHA-256 manifest metadata.

NOAA files commonly contain multiple reports at the same timestamp (for
example FM-12 and FM-15). Keeping only the last row can discard the valid
pressure value. SkyGuard keeps duplicates through parsing and performs
field-wise hourly median aggregation. This is a correctness requirement, not
an optional cleanup.

For NOAA, RH is calculated from temperature and dew point using the Magnus
relationship. It must be described as a derived channel, not an independent
RH sensor. NOAA SLP is retained as `pressure_hpa` with explicit provenance;
downstream claims should not call it station/barometric pressure unless the
source field supports that interpretation.

## Labels and evaluation claims

Open public weather archives do not provide a complete timestamped maintenance
log with a consistent nine-class sensor-fault taxonomy. The project therefore
injects physically constrained faults into clean, real station observations.
The labels are useful and reproducible, but they are not field-confirmed IMD
fault labels. Reports must say **“performance on injected faults over real
weather”** and must include strict point-wise F1, range-wise F1, PR-AUC,
per-fault recall, latency, and station-level holdout results. Point-adjusted F1
is supplementary because it can inflate scores for long anomaly ranges.

The split policy is chronological (train before 2023, validation 2023, test
2024 onward where available) plus a deterministic 20% station holdout. The
holdout is excluded from global model fitting, ensemble-weight calibration,
and supervised thresholds. Station-local scaling and a train-only POT fallback
are reported as unsupervised adaptation; this is not a zero-history cold-start
claim.

## Reproducibility and licensing

The processed Kaggle bundle must carry `docs/LICENSES.md`, the source URLs,
parser version, split metadata, and a content-addressed `MANIFEST.json`.
IMD-DSP and NCPOR files remain manual/private inputs unless their terms permit
redistribution. Open-Meteo attribution and its terms must travel with any
derived bundle. NAB's MIT license must remain with the benchmark files.

## Acceptance checklist before quoting a result

1. Rebuild the online bundle after any parser or feature change; do not quote
   metrics from the stale local Parquets.
2. Verify that `splits.json` group counts are dictionaries with `train`, `val`,
   and `test`, and that all headline groups have complete coverage.
3. Train headline metrics on `source=noaa_isd` only. Report Open-Meteo and NAB
   in separate reference sections.
4. Preserve the exact bundle manifest, source roles, station holdout list, and
   `artifacts/metrics.json` with the submission.
