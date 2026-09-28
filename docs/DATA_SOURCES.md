# Data sources — verification record (checked 2026-09-16/17)

This is a historical source record, not a current verification certificate.
The [2026-09-23 real-data audit](REAL_DATA_RESEARCH.md) supersedes its acceptance
decisions. A published field or provider quality flag does not by itself prove
an independently measured AWS channel or a confirmed sensor fault.

## 1. NOAA Integrated Surface Database (ISD, "global-hourly") — PRIMARY
- What: hourly/sub-hourly surface observations from real stations feeding the
  global telecom network. For India this is an AWS-compatible surface-station
  proxy; NOAA metadata alone does not prove that every station is an IMD AWS
  installation.
- Endpoint (verified, HTTP 200, CSV):
  `https://www.ncei.noaa.gov/data/global-hourly/access/<YEAR>/<USAF><WBAN>.csv`
  e.g. `.../2024/42182099999.csv` → Safdarjung (New Delhi), 7.9 MB.
- Station catalogue: `https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv`
  (2.9 MB, verified; 425 Indian stations with data through Aug 2025).
- Fields used: TMP (×0.1 °C), DEW, SLP (×0.1 hPa), WND, AA1 (precip), QC flags.
- Volume obtained: 21 curated climate-spanning stations × 2018–2025 = 168 files, 678 MB.
- Notes: 3-hourly synoptic cadence for non-airport stations; quality codes 0/1
  kept; missing markers (9999 etc.) honoured. A few stations/years lack the AA1
  column — parser handles absent columns.

## 2. Open-Meteo Historical Archive API — co-located reanalysis twins
- Endpoint: `https://archive-api.open-meteo.com/v1/archive` (CSV format verified).
- Variables: temperature_2m, relative_humidity_2m, dew_point_2m, pressure_msl,
  surface_pressure, wind_speed/direction/gusts, precipitation, shortwave.
- Obtained: 21 stations × 2019-01-01..2025-12-31 hourly (ERA5-seamless), 78 MB.
- One station returned a server-side corrupt stream with HTTP 200 (+error text);
  the downloader validates the payload and refetches with the default model mix —
  caught and repaired for Srinagar (420270).
- Used as: independent co-located reference for the spatial/buddy-check and as
  an additional distribution to learn from.

## 3. Numenta Anomaly Benchmark (NAB) — external labeled benchmark
- Repo: `https://github.com/numenta/NAB` (clone verified; data + labels included).
- NAB's own data README describes *ambient_temperature_system_failure* as
  ambient temperature in an office, not a documented AWS sensor failure.
  Its industrial-machine and CloudWatch series are also not AWS weather
  ground truth. The corpus includes artificial collections; those are excluded
  by the revised real-data requirement.

## 4. NCPOR meteorological repository (MoES) — IMD AWS in polar regions
- `https://data.ncpor.res.in/newhtml/` — verified live catalogue:
  - Maitri AWS (IMD): 1985–2016, hourly: temp, pressure, wind, RH
    (`/newhtml/download/70`, dataset metadata `/static/datasets/m_imd_aws.txt`)
  - Bharati AWS (IMD/IIG): 2012–2016+ (`/newhtml/download/75`, `/78`)
  - Himalayan data section exists (`/newhtml/` category 5).
- **Access note**: download endpoints sit behind a CAPTCHA page (verified by
  header/content check). Manual step: open the page in a browser, solve CAPTCHA,
  save authorised files into `data/raw/ncpor/`. A source-specific parser still
  needs to be implemented; the current loader does not ingest NCPOR files.
  The 2026-09-23 inventory found this directory empty.

## 5. IMD Data Supply Portal (official channel)
- `https://dsp.imdpune.gov.in` — verified live; requires enrolment and, for
  non-MoES users, payment for historical data. This is the authoritative path
  for *additional* Indian AWS hourly logs beyond the open sources above.
- Not required for the pipeline to function end-to-end.

## 6. imdlib (PyPI) — gridded IMD rainfall/temperature
- Package exists (pip). Gridded products (0.25°) rather than station streams —
  optional climate context; not needed by default.
- PyPI page fetch is bot-gated; obtained via local install instead.

## 7. NOAA GSOD (daily summaries) — supplementary
- `https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/<YEAR>/` —
  verified directory listing (1929–2025). Daily aggregation is too coarse for
  hourly anomaly detection but usable for long-term drift checks.

## 8. NASA SMAP/MSL labeled telemetry (transfer benchmark)
- Repo: `github.com/khundman/telemanom`; data distributed as the Kaggle dataset
  `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`. On Kaggle: just attach
  it alongside ours (no upload needed); the notebook auto-detects it.

## What we deliberately did NOT use
- Kaggle "weather csv" hobbyist dumps of unknown provenance — rejected for
  authenticity; our reproducible base comes from NOAA ISD surface-station
  feeds, with IMD-DSP/NCPOR reserved for authorised external AWS validation.
- MOSDAC (registration-gated; needs manual account; follow-up only).
