# SkyGuard AI — SIH26073

Minute-level anomaly candidates from **original measured temperature, station
pressure and relative humidity**. The active pipeline uses NOAA SURFRAD native
one-minute observations, preserves missing data and source quality codes, and
never invents readings or fault labels.

The detector combines one-minute and one-hour forecast residuals, abrupt changes,
exact-repeat duration and sustained deviations. Outputs describe suspicious
behaviour for review; they do not diagnose hardware causes or repair observations.

The completed local run processed **2,836,536 development observations** and
replayed **133,920 fresh January 2025 observations** using frozen models and
thresholds. All 56 original columns matched the verified native inputs exactly.
**228 tests and 6 subtests passed**, including the real dashboard.

**Real hardware-fault accuracy remains unmeasured.** The fresh replay produced
170 candidate intervals, not 170 confirmed failures. The reviewed official
records do not establish fault labels for this 2023–2025 scope.

- [Completed minute experiment and reproduction](docs/MINUTE_DETECTION_20260929.md)
- [Official event-evidence research](docs/research/surfrad_events_20260928/VERIFICATION.md)
- [Source verification](docs/research/surfrad_20260926/VERIFICATION.md)
- [Real-data rules](docs/REAL_DATA_RESEARCH.md)

## Historical replay dashboard

```powershell
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The dashboard defaults to `artifacts_minute_20260928`. Enter
`artifacts_minute_fresh_20260928` in its sidebar for the January replay.
The **USA · Real observations** page shows original readings, separate
predictions, alert reasons, gaps, provider QC and raw-file/row provenance.

The separate **India · Synthetic scenarios** page provides the user-requested
Indian demonstration based on archived Indian NOAA ISD reports. It compares
unchanged source values with explicitly synthetic scenarios. Calculated RH and
sea-level pressure keep their actual meanings. These scenarios are excluded
from the real-data training and accuracy claims. See the
[Indian station demo guide](docs/INDIAN_STATION_DEMO.md).

## Reproduce native training

Use the existing verified raw archive or acquire it with
`python scripts/fetch_real_surfrad.py`. Then use a new output directory:

```powershell
python scripts/run_minute_detection.py --archive-dir data/raw/surfrad_original --cache-dir data/native_minute_20260928 --out artifacts_minute_new
```

The default protocol fits on Bondville/Fort Peck before July 2024, selects models
in July–August, calibrates in September–October, and excludes Goodwin Creek from
all fitting, selection and calibration. Previously examined November–December
2024 test data are excluded. January 2025 has now been evaluated and must not be
reused for further selection while calling it an untouched test.

## Kaggle workflow

Keep the existing **source dataset → builder notebook → processed dataset**
observation flow. The new `kaggle/aws_minute_detection.ipynb` uses the originals
already retained in the processed release and embeds the exact updated minute
source with integrity checks. `kaggle_minute/` is ready for upload; CPU is enough.

```powershell
python kaggle/build_minute_notebook.py
```

The notebook is generated and validated locally; it has **not been published or
run on Kaggle**. The previously completed hourly training v14 remains unchanged.
See [the Kaggle guide](docs/KAGGLE_GUIDE.md).

## Evaluate independently reviewed events

Review the exported `review_template.csv` using actual maintenance, calibration,
operator records or independent measurements. Leave unsupported cases unknown.

```powershell
python scripts/evaluate_real_events.py --artifacts artifacts_minute_fresh_20260928 --reviews artifacts_minute_fresh_20260928/review_template.csv --out reviewed_evaluation.json
```

The evaluator requires evidence references and a reviewer for accepted labels.
It reports event recall/delay and false alarms only within independently reviewed
intervals; unknown cases never become negative examples. Frozen single-signal
comparisons are available via `--signal` on the same reviews.

The former injected-data design and commands are preserved in
[the historical README](docs/LEGACY_README.md) for reference. They are excluded
from the current real-only workflow and accuracy claims.
