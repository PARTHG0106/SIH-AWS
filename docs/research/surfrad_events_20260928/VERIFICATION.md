# Verified research into real SURFRAD event evidence

Research: 28–29 September 2026. Scope: Bondville (`bon`), Fort Peck (`fpk`),
Goodwin Creek (`gwn`), 2023–2025, measured ambient temperature, relative humidity
and station pressure.

**No independently documented, usable event labels or reviewed fault-free
intervals were established for this scope.** This does not mean there were no
faults. It means the inspected evidence does not support fault precision,
recall, F1, false-alarm or detection-delay claims for the current replay.

The research did establish five historical incident records relevant to these
channels, including one old RH incident whose retained bad readings were checked
in two actual provider files. They are documented below rather than silently
turned into labels for a different period or sensor generation.

The [machine-readable registry](event_evidence_registry.json) contains the
source quotations, time uncertainty, archive corrections and exclusion reasons.
The [source manifest](source_manifest.json) preserves requested/final URLs,
retrieval times, HTTP metadata, immutable response SHA-256 hashes, receipt hashes
and local paths. It covers **23 successful downloads and one HTTP 404**. The
original 21 downloads were obtained on 28 September; the two old daily files
were obtained on 29 September. The failed request used an incorrect USCRN
filename; the correct filenames were then read from the downloaded directory
listing and fetched successfully.

## Historical incident evidence

All five records below come from NOAA's [SURFRAD Network Data Problem
Reports](https://gml.noaa.gov/grad/surfrad/problems.html), retrieved
`2026-09-28T11:48:16.075258+00:00`, SHA-256
`81e1fadfbef2259cc38b4aa0d4c4d72c6ad95dc2a06df02efd2b59ea11c41355`.
[Saved original](downloads/81e1fadfbef2259cc38b4aa0d4c4d72c6ad95dc2a06df02efd2b59ea11c41355.html).

| Station and provider dates | What the provider actually documents | Why it is not a current benchmark label |
|---|---|---|
| Goodwin Creek, July 1997–2 June 1998 | A replacement RH chip was faulty and its readings were erroneously low. The provider says the values were retained and assigned bad QC in the reprocessed archive. | Outside 2023–2025; native three-minute data and an older probe. The table starts on **20 July**, but the narrative says **10 July**. No exact onset or recovery time is established. |
| Bondville, 4 June–22 July 1999 | Lightning destroyed the barometer and T/RH probe and damaged logger channels. All data after the strike through 7 June were lost. A follow-up says the problems were fixed during the 20–22 July exchange. | Outside scope; no affected raw files were acquired for this incident. A missing reading is not a numerical fault example. The 26 July follow-up date is not used as an exact repair timestamp. |
| Goodwin Creek, 16 July–26 August 2004 | A lightning-caused short in the T/RH probe drained logger power, preventing logging and communication. Replacing the probe restored collection on 26 August. | Outside scope; event measurements not acquired. This is documented T/RH-probe damage and a station outage, **not a diagnosed pressure-sensor failure**. Availability evaluation needs the appropriate raw logging evidence. |
| Penn State, 11 April 2003–18 June 2012 | Barometer drift of about −4 mb across the period. | Outside time and station scope. NOAA explicitly corrected **all** affected pressure data and reprocessed the daily files. The current archive cannot be assumed to contain the original drift. Uncorrected originals were not acquired. |
| Penn State, 2016–19 September 2017 | A wrong logger equation converted the barometer signal to pressure, producing about −10 mb error. | Outside scope; this is a conversion/program error rather than demonstrated barometer hardware failure. All affected files/products were regenerated with corrected pressure. The table and reprogramming date say **13 July 2016**, while the narrative parenthesis says **13 September 2016**. |

The saved problem page's first notice concerns UVB in 2018–2019, posted in June
2019. There are no `2023`, `2024` or `2025` entries in the downloaded page. This
is a limit of the inspected public page, not an assertion that all maintenance
records have been searched or that none exists privately.

Other notices describe pyrgeometer case/dome thermistors, swapped radiation
instrument connections, corrected irradiances and interpolated soundings. Those
temperatures are not the ambient `temperature_c` channel. General Campbell
Scientific 207 RH degradation and CS500 replacement dates provide instrument
history, but do not identify additional reviewed per-station failure intervals.

## An actual old RH case was checked in the archive

The 1997/1998 Goodwin directory listings contain links to `gwn97202.dat` and
`gwn98152.dat`. Both were fetched without changing their bytes. The provider's
README documents 48 fields, RH at field 41 followed by its QC code, UTC times
and three-minute averages before 2009. The parser used for this research is in
[audit_evidence.py](audit_evidence.py); it does not use the one-minute training
adapter or write a training dataset.

| Actual file date | Rows | Actual adjacent cadence | Nonmissing T/RH/pressure | RH raw QC | RH range |
|---|---:|---:|---|---|---|
| 21 July 1997 | 480 | 180 seconds throughout | 480 / 480 / 480 | `1` on all 480 rows | 58.6–82.4% |
| 1 June 1998 | 480 | 180 seconds throughout | 480 / 480 / 480 | `1` on all 480 rows | 16.0–62.0% |

These are observations during the independently documented old RH incident.
Their ranges alone do not prove failure, and the QC flags alone do not establish
the hardware cause. The **provider's incident narrative** supplies the cause
evidence. This corroborates the statement that bad RH values remain in the
archive on these two days; it does not certify every record across the full
incident interval or a fault-free comparison period.

The samples' SHA-256 hashes are
`a45967fb18ecc9c236d19f4021acf9e8cd58ae16adf5fbee61ed9c24bf4280a4`
and `016560ca5c2c987d2d2caa223fd2f49771a289b7b1b8c3e79fea2efc84ba366f`.
The complete acquisition metadata is in the manifest. These files remain
**research evidence only**. A separate historical RH case study could be useful
after review of the full interval and original lineage; a detector expecting
one-minute records cannot be tested by manufacturing intermediate records.

## Calibration records checked

The saved [Bondville](https://gml.noaa.gov/grad/cgi-bin/surfcals?site=Bondville),
[Fort Peck](https://gml.noaa.gov/grad/cgi-bin/surfcals?site=Fort%20Peck), and
[Goodwin Creek](https://gml.noaa.gov/grad/cgi-bin/surfcals?site=Goodwin) CGI tables
were inspected. Their headers list downwelling/upwelling pyranometers and
pyrgeometers, UVB, PAR, normal-incidence pyrheliometers and shaded pyranometers.
**They contain radiation-instrument serial numbers and calibration constants,
not T/RH/pressure service records.** Recent exchange dates are not sufficient
evidence of a failure in one of the three input channels.

## Independent nearby observations checked

The NCEI [USCRN station table](https://www.ncei.noaa.gov/pub/data/uscrn/products/stations.tsv)
and [Hourly02 documentation](https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/README.txt)
were downloaded and read. Two actual 2024 files were then inspected:

| USCRN station | Location evidence | Hourly rows | Present hourly temperature / RH |
|---|---|---:|---:|
| Champaign 9 SW, WBAN 54808 | Station table names the Bondville research station at 40.05, −88.37; SURFRAD Bondville page gives 40.05192, −88.37309. | 8,784 | 8,776 / 8,779 |
| Wolf Point 29 ENE, WBAN 94060 | Station table gives Poplar River Site at 48.3, −105.1; SURFRAD Fort Peck gives 48.30783, −105.10170. | 8,784 | 8,775 / 8,776 |

Coordinates identify nearby locations; their rounding does not establish exact
co-location, common exposure or instrument serial identities. Hourly02 temperature
is a provider aggregate from multiple independent measurements, and RH is an
hourly aggregate of five-minute averages. These are **not** a one-minute
replacement input or direct evidence that a SURFRAD sensor failed. The product's
complete 38-field schema has **no pressure field**; `P_CALC` is precipitation in
millimetres, not pressure. The hourly timestamps mark the end of the averaging
period. Both files end at `2025-01-01 0000`, representing the final hour of 2024.

All 8,784 RH QC codes in each file are `0`, including rows whose RH is missing.
Missing values therefore still need their own check. Neither missing values nor
QC=0 were converted into background/normal truth. No paired discrepancy review
was performed, and no fault labels were assigned from these samples. The
inspected files/documentation establish useful context availability; they do
not establish a complete redundant-sensor benchmark or redistribution terms
for a new USCRN training release.

## Review workflow and reproducibility

Detector-generated candidate intervals and sampled non-alert intervals are
**review proposals**, each initially `unknown`. A review that enters evaluation
must carry a station|source group, channel, UTC half-open interval, named
reviewer, review time, independent evidence type, source URL and immutable hash.
Confirmed events also need onset uncertainty bounds. A human/domain reviewer
must check what the cited bytes actually support; passing the schema validator
does not certify a claim.

`awsad.evaluation.real_events` excludes unknown intervals, rejects overlapping
accepted reviews, and reports event recall and bounded delay only over reviewed
events. It reports false alerts only in explicitly reviewed background and
does not count gaps as monitoring exposure. Alert-selected review samples do
not estimate population precision or accuracy. An incident with no scored
observations is explicitly reported, rather than silently removed from recall.

After independent review, [evaluate_real_events.py](../../../scripts/evaluate_real_events.py)
consumes a reviewed CSV and the replay artifact directory. It verifies each
scored shard against the saved SHA-256 manifest, preserves unknown counts,
joins reviewed intervals across monthly shards and records the review-file and
evaluation-code hashes. Use a new output file for each evaluation:

```powershell
.venv/Scripts/python.exe scripts/evaluate_real_events.py --artifacts ARTIFACT_DIR --reviews REVIEWED_CSV --out NEW_EVALUATION.json
```

Repeat with `--signal abrupt_change`, `--signal flatline_minutes` or
`--signal forecast_residual` and separate output paths to compare those saved
signals against the combined detector on the **same** reviewed intervals. The
thresholds are read from the frozen detector; they are never retuned by this
command. With only `unknown` reviews, fault metrics remain null. All 24 event
evaluation and CLI tests passed using test-only fixtures; this validates software
behaviour, not real-world detector accuracy.

Recheck the archive hashes, receipt hashes, all 12 quoted passages and measured
sample counts with:

```powershell
.venv/Scripts/python.exe docs/research/surfrad_events_20260928/audit_evidence.py
```

The audit uses only saved files and never downloads, edits observations or
creates labels. The SURFRAD README explicitly documents CC0 1.0 for NOAA's
internally published data and requests the Augustine et al. (2000, 2005)
citations. Preserve that attribution when using the SURFRAD observations.

Useful next evidence would be actual 2023–2025 T/RH/barometer maintenance,
calibration or operator records linked to these stations, or independently
reviewed measurement comparisons with matching time support and documented
sensor identity. No message was sent to a provider. Until that evidence is
obtained, current detector outputs remain unconfirmed candidates.
