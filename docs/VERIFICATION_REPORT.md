# Verification report

Checked on 2026-09-17. This record separates locally verified evidence from
sources that remain registration- or CAPTCHA-gated. The byte counts below are
the original curated download audit; the reproducible Kaggle builder can now
extend the NOAA station set without changing the detector contract.

## Problem statement

The implementation targets the user-supplied SIH record `SIH26073`, titled
"AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations",
for MoES/IMD under Disaster Management. The official SIH page was not archived
inside the workspace, and browser access to search it was unavailable during
the final audit. The ID/title/agency should therefore be cross-checked once
against the current official SIH portal before submission.

## Locally verified datasets

| Dataset | Evidence in workspace | Role | Important limitation |
|---|---:|---|---|
| NOAA ISD Global Hourly | 168 station-year CSVs, 711,594,924 bytes; 21 Indian stations for every year 2018-2025 | Primary real station observations | ISD provides dew point and sea-level pressure here; RH is derived from T/Td, not a directly reported RH sensor channel |
| Open-Meteo Archive | 21 raw hourly CSVs, 86,182,929 bytes | Co-located reanalysis reference and distributional diversity | Reanalysis is not AWS sensor ground truth and must not be presented as observed sensor data |
| Numenta NAB | 58 CSV series, 9,932,226 bytes, MIT license present | External streaming-method sanity benchmark | Mostly non-weather, univariate data; it does not validate multivariate AWS fault accuracy |

Representative SHA-256 checksums:

```text
data/raw/noaa_isd/isd-history.csv
1994747AB4AF1B97E63ADB434B4D0D022F2DAEE76F0C144EA9AB46BE2D906604

data/raw/noaa_isd/2024/42182099999.csv
E54A28AF2334907E83670210345A49565E9443EB0A5B935B3AA1095A6B3E9356

data/raw/openmeteo/42182099999.csv
AC0653DE040566CD03A78B26B39032D9CD780AAA05B8894424FE664671368A5F

data/raw/nab/NAB/LICENSE.txt
930B48879944CE3156E2B1AFCF89A582B3C26098463350614580F0FFBFDA9275
```

The processed dataset contains 1,656,444 clean training rows, 367,920
validation rows, and 671,748 test rows. Injected anomaly labels cover 12,267
validation rows and 20,956 test rows (about 3% after event overlap).

## Source suitability decisions

- NOAA ISD is the strongest openly reproducible station-level source in this
  workspace and is the primary model foundation.
- Open-Meteo is retained as a separate `source` group and as a buddy reference.
  It is never described as a physical AWS observation.
- IMD-DSP is authoritative but access-controlled; it is a production data
  onboarding path, not a reproducible dependency of this repository.
- NCPOR contains relevant MoES/IMD polar AWS series, but the download path is
  CAPTCHA-gated. No NCPOR files are bundled.
- `imdlib` and IMD gridded products are useful climate context but are not
  station sensor streams, so they are not used as anomaly labels.
- NOAA GSOD is daily and too coarse for the hourly detector.
- NASA SMAP/MSL is a transfer benchmark only. It is not weather data and is not
  currently bundled in this workspace.

## Label validity

Public weather archives generally do not provide a complete, timestamped fault
taxonomy. The project therefore injects nine reproducible sensor-fault types
into real, chronologically held-out weather observations. Accuracy claims must
say "on injected faults over real weather" rather than implying that every
label came from an IMD maintenance record.

The built corpus writes a deterministic station-level holdout set to
`data/processed/splits.json` (20% of stations with complete train/validation/
test coverage by default, while retaining the four legacy preferred stations
when they are present). Every matching `station|source` group is excluded from
global model fitting and label-based calibration. Their station-local
climatology, robust scale, and POT threshold use only clean historical data.
This is unsupervised adaptation, not zero-history cold-start evaluation, and is
reported as such. Older checked-in snapshots may still contain the historical
21-station/4-holdout metadata; rerun the builder before quoting corpus counts.
