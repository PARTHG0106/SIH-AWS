# Data licenses and attribution

## Current real-observation training source (2026-09-26)

The migrated builder acquires NOAA Global Monitoring Laboratory **SURFRAD**
daily station observations. The official `README_SURFRAD.txt`, inspected and
archived in `docs/research/surfrad_20260926`, expressly releases SURFRAD data
under **CC0 1.0**. Retain the provider documents and original acquisition
receipts with redistributed data. Acknowledge NOAA GML and cite Augustine,
DeLuisi and Long (2000), BAMS 81, 2341–2357, and Augustine et al. (2005),
Journal of Atmospheric and Oceanic Technology 22, 1460–1472, as requested in
that README. Do not imply NOAA endorsement. Data are provisional and may be
revised by the provider; this project preserves acquired bytes by SHA-256.

The source-code bundle also contains project code and historical research
documents, so the whole ZIP is not relicensed CC0. The entries below describe
legacy or prospective sources, not inputs used by the migrated training run.

The Kaggle bundle combines material with different terms. It must not be
described as a single CC0 dataset.

## NOAA Integrated Surface Database

- Provider: NOAA National Centers for Environmental Information (NCEI)
- Source: https://www.ncei.noaa.gov/data/global-hourly/access
- Status: United States government data are generally public domain. NOAA asks
  users to cite NCEI and provides the data without warranty.

Suggested attribution: "NOAA National Centers for Environmental Information,
Integrated Surface Database (Global Hourly)."

## Open-Meteo historical archive

- Provider: Open-Meteo, backed by the weather models named by the API response
- Source: https://archive-api.open-meteo.com/v1/archive
- Terms: https://open-meteo.com/en/terms

Open-Meteo requires attribution. The raw CSV metadata and this repository's
source record must remain with redistributed processed data.

Suggested attribution: "Weather data by Open-Meteo.com."

## Numenta Anomaly Benchmark

- Source: https://github.com/numenta/NAB
- License: MIT
- The original `LICENSE.txt` is copied to `nab/LICENSE.txt` in the bundle.

## IMD and NCPOR data

IMD Data Supply Portal and NCPOR files are documented as optional sources but
are not redistributed in the generated Kaggle bundle. Teams adding those files
must comply with the terms attached to their request or download.

## SkyGuard source code

No project-wide open-source license has been selected in this repository.
Copyright remains with the project authors until they add an explicit license.
