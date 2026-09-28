# Read-only Kaggle metadata audit — 2026-09-25

Retrieved at **2026-09-25T11:46:45Z** using the authenticated Kaggle API.
No observation/parquet files were downloaded. No kernels were started and no
dataset versions were published. API credentials and error bodies are absent
from the saved evidence.

## Verified release chain

| Item | Live version / identity |
|---|---|
| Source dataset | `krishnagupta02468/skyguard-sih26073/17`, ready, 40 listed files |
| Source manifest SHA-256 | `0e7c7c1e3699483df319a559441c0a80194a24ff772a67a7a05122ddf21ef134` |
| Processed dataset | `krishnagupta02468/skyguard-sih26073-processed/8`, ready, 120 listed files |
| Processed build ID | `20260925T105454Z-0e7c7c1e3699` |
| Processed build timestamp | `2026-09-25T10:54:54.246517+00:00` |
| Processed manifest SHA-256 | `74a563e6f77674f296a0f979833e88f2e369234bb730ed21fd9a2a21ee7307be` |

The processed manifest, `source_metadata.json`, and `splits.json` all contain
the same build ID and source-manifest hash. The source-manifest hash matches
the bytes downloaded from source version 17. The processed manifest's hashes
for `data/processed/source_metadata.json` and `data/processed/splits.json` match
the downloaded bytes. Its hash for `provenance/source_manifest.json` also
matches the downloaded source manifest.

This is exactly the build ID in the user's latest builder log and is newer
than the original 2026-09-24 training run. The Kaggle creation timestamps for
the processed files are 2026-09-25 11:19–11:20 UTC. These checks verify the
release metadata chain; they do not independently verify the contents of the
large observation files, which were deliberately not downloaded.

## What the new metadata reports

The fresh release still explicitly describes the legacy data policy:

- 107,872 synthetic positive points in validation (96.87% of positive points)
  and 161,622 in test (96.32%).
- NOAA provider QC flags supply 3,486 validation and 6,173 test positives.
  Provider QC is quality evidence, not an independently confirmed hardware
  fault diagnosis.
- Between 36.7% and 39.2% of the three channels' readings, depending on split
  and channel, are reported as imputed and retained. The configured maximum
  gap fill is six hours.
- The metadata says 100% of NOAA RH is calculated from temperature/dew point.
- The pressure policy explicitly uses NOAA SLP and Open-Meteo pressure_msl,
  which are sea-level pressure.
- It reports 422 stations, 443 station/source groups, and 293 groups meeting
  the legacy headline-eligibility contract. These counts are release claims,
  not independently measured three-sensor observation coverage.

Freshness therefore does not establish compliance with the project's binding
real-only observation and label requirement.

## Evidence files

- `audit.json`: pinned remote references, complete file inventories, download
  sizes, retrieval timestamp, and SHA-256 hashes.
- `download_1/MANIFEST.json`: unchanged source-v17 manifest, 5,118 bytes.
- `download_2/MANIFEST.json`: unchanged processed-v8 manifest, 158,637 bytes.
- `download_3/source_metadata.json`: unchanged processed metadata, 3,375 bytes.
- `download_4/splits.json`: unchanged processed split metadata, 402,124 bytes.

The read-only reproduction script is
`scripts/audit_kaggle_processed_metadata.py --download`. It allows only the
three named metadata basenames, requires a listed file size no larger than
2 MB, pins the selected versions, and stores downloaded metadata separately
from the training dataset.
