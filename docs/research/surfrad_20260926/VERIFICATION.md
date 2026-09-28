# SURFRAD source verification — 26 September 2026

The inspected NOAA documents and acquired daily-file samples support a native
observation view of **Bondville (`bon`), Fort Peck (`fpk`), and Goodwin Creek
(`gwn`), 2020–2025**. This admits provider-published in-situ temperature, relative
humidity and station-pressure measurements outside the corrections listed in
the inspected problem report. It does not certify fault-free readings, hardware
fault labels, unobserved station coverage, or an Indian AWS validation corpus.
The requested acquisition is the three sites in **2023–2024**; actual coverage
must come from the downloaded directory listings and raw files, not a presumed
number of days or observations.

## Inspected official evidence

The retrieval receipts preserve the requested/final URL, actual UTC retrieval
time, byte count and SHA-256. These hashes identify the exact reviewed versions;
changed provider documents require another source review.

| Evidence | Official URL | SHA-256 |
|---|---|---|
| Daily format, QC, units, native averaging and licence | <https://gml.noaa.gov/aftp/data/radiation/surfrad/Bondville_IL/README_SURFRAD.txt> | `ed3e5c16f169a9def997877e6037735977f2761c9c3c39d493c8f3127e82f3a6` |
| Network instruments | <https://gml.noaa.gov/grad/surfrad/overview.html> | `c0f29ff21e4bd2d2c05a679a46435fdc627067f9f4280f2b8a00063c861f60fc` |
| Published corrections and failures | <https://gml.noaa.gov/grad/surfrad/problems.html> | `81e1fadfbef2259cc38b4aa0d4c4d72c6ad95dc2a06df02efd2b59ea11c41355` |
| NOAA terms and attribution | <https://gml.noaa.gov/about/disclaimer.html> | `657e14026cea7e71431d1fbd1920d2d671bbd289009bb8e3ed8bb26e17ea7dec` |

Copies are in this directory and its `downloads/` subdirectory. Acquisition
receipts and the finalized source bundle also preserve these documents.

## Measurement and format evidence

The network overview explicitly describes instruments measuring air temperature
and relative humidity on a **10 m tower**, a replaceable primary RH sensing chip,
and a barometer in the logger enclosure measuring **station pressure**. This
establishes a humidity-sensor channel independent of a temperature/dew-point
calculation. The inspected pages do not provide complete model/serial-number
histories for 2020–2025; those attributes must remain unknown.

Historical instrument information is explicit but must not be extrapolated to
the modern period: the first four stations used Campbell Scientific 207 T/RH
sensors. The provider reports CS500 replacements at Table Mountain on
1997-08-26, Fort Peck on 1997-09-23, Bondville on 1998-06-19 and Goodwin Creek
on 1998-06-02. Desert Rock and Penn State originally used CS500 probes.
Pressure instruments were added during 1996–1997; earlier pressure placeholders
cannot be treated as measurements.

For the **48-field daily observation format**, whitespace-separated zero-based
positions are:

| Numeric input | Value position | QC position | Raw name/unit |
|---|---:|---:|---|
| `temperature_c` | 38 | 39 | `temp`, degrees Celsius |
| `relative_humidity_pct` | 40 | 41 | `rh`, percent |
| `pressure_hpa` | 46 | 47 | `pressure`, millibars; 1 mb = 1 hPa |

The first header is the station name. Actual acquired Bondville files use a
second header such as `40.05 -88.37 213 m version 1`: six tokens including
literal `m version`, followed by the provider file version. The parser was
checked on **637 acquired daily files**, returning **15,142 exact-hour rows**
without parsing errors in that acquisition snapshot. This is a sample check,
not the final three-site coverage result. A later header inventory covered
1,177 files (710 Bondville, 467 Fort Peck). As Goodwin Creek files arrived,
the first published day of both 2023 and 2024 was parsed for all three sites;
each of those six samples retained 24 actual minute00 records. Fort Peck uses
`48.31 -105.10 634 m version 1` and Goodwin Creek uses
`34.25 -89.87 98 m version 1`; both match the observed six-token structure.
The completed bundle's verifier must
reparse **every** acquired daily file and compare every retained numeric value,
raw string, timestamp, source code, QC code, identifier, missing value and label
with the processed table.

Since 2009-01-01, the native reports are **one-minute averages of one-second
samples**. Earlier reports were three-minute averages. Timestamps are UTC and
denote the **end** of the native averaging interval; `00:00` therefore ends an
interval beginning on the preceding day. Selection keeps only existing
`minute == 0` records. It does not average an hour, interpolate, resample to a
complete grid, or create a row when minute 00 is missing. Native file bytes,
original physical line numbers, headers and field strings are retained.

The documented missing sentinel is `-9999.9`, normally accompanied by QC 1.
The numeric view uses a missing value while retaining the original sentinel and
flag. NOAA states that bad data are deleted and questionable data flagged;
missing time periods can be absent entirely. Their absence must remain visible.

QC 0 means the reading passed the provider's checks; QC greater than zero means
at least one QC level failed. A present value with QC 0 versus a present value
with QC greater than zero can support **provider-QC agreement**. Missing values
have no such target. These are not reviewed normal/fault hardware labels. Every
hardware-fault label in this acquisition remains nullable and **unknown**.

## Corrections and exclusions

The complete inspected problem page was checked for T, RH, station pressure,
timestamps and changes affecting all fields. The following records are relevant
to these inputs. All fall outside the accepted 2020–2025 scope, so no corrected
values from those periods enter this release.

| Site / field | Published period | Evidence and treatment |
|---|---|---|
| Penn State / station pressure | 2003-04-11 through 2012-06-18 | A linear drift correction was applied and daily files regenerated. Exclude corrected pressure in this interval. |
| Penn State / station pressure | 2016-07-13 through 2017-09-19 | Wrong raw-signal conversion equation; pressure recomputed. The table says July 13 while part of the prose says September 13. Use the earlier July boundary conservatively; do not assert the conflict resolved. |
| Table Mountain / all timestamps | 2012-10-23 through 2013-05-05 | Logger clock drift; linear time corrections applied to raw files and products. Exclude the affected timestamps rather than presenting corrected times as original. |
| Goodwin Creek / RH | 1997-07-10 through 1998-06-02 | Faulty RH chip produced low readings, which were retained but marked bad. Header says July 20; prose identifies July 10. Use the earlier date for exclusion/review. No automatic cause labels are created. |
| Bondville / T, RH, pressure and other fields | 1999-06-04 through 1999-07-22 | Lightning destroyed the T/RH probe and barometer and caused data loss/logger damage. Repairs were completed during July 20–22. Do not manufacture readings for the lost interval. |
| Goodwin Creek / all fields | 2004-07-16 through 2004-08-26 | Lightning-induced short in T/RH probe drained logger power and stopped logging. Do not fill the outage or assign unreviewed per-row fault labels. |
| All sites / native sampling | 2009-01-01 onward | Files were reprocessed from parallel **actual one-minute raw data**, replacing the earlier three-minute output. This is native sample availability, not interpolation. The parser supports the verified modern format only. |

The README and problem page additionally describe calculated/reprocessed solar
and infrared fluxes, UVB assumptions and interpolated sounding products. In
particular, Fort Peck pyrgeometer **case temperature** was artificially generated
for 2000-12-10 through 2001-07-24. This is not the meteorological `temp` field.
None of the radiometric fields, pyrgeometer temperatures, net-flux calculations,
UVB/PAR corrections, QcRad3/RadFlux outputs or interpolated soundings supplies a
required input in this dataset. No later T/RH/pressure/time correction covering
the admitted scope was listed in the inspected page. This is a statement about
the reviewed published evidence, not a guarantee that future revisions or
undiscovered problems cannot exist.

## Rights and limits

The product-specific README explicitly licenses NOAA-produced SURFRAD datasets
under **CC0 1.0 Universal**. The NOAA disclaimer permits public use, requires
honest attribution, prohibits implying NOAA endorsement or presenting modified
material as official NOAA output, and requests acknowledgement of NOAA Global
Monitoring Laboratory. Retain that acknowledgement and the README's suggested
Augustine et al. (2000, 2005) citations. These US environmental monitoring sites
are not a substitute for independent validation on Indian IMD AWS installations.

DWD was also checked using live official descriptions, terms, station00433 ZIPs
and instrument histories. It was **not admitted**: the sampled 2024 hourly
exports have QN3 (automatic control and correction) without per-value correction
flags, and independently measured exported RH origin remained unresolved.
See [DWD verification](../dwd_20260926/dwd_verification.json) and its immutable
retrieval manifest. DWD files remain research evidence only.
