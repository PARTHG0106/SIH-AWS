# SkyGuard AI — SIH26073

An operational prototype for **temperature, station pressure and relative
humidity** anomaly review. The real-observation path uses NOAA SURFRAD native
one-minute measurements, preserves missing data and source quality codes, and
never fabricates observations or real hardware labels. A separate, explicitly
synthetic scenario pipeline trains and evaluates software-pattern recognition,
as permitted by the [official SIH26073 statement](https://www.sih.gov.in/sih2026PS).

The detector combines one-minute and one-hour forecast residuals, abrupt changes,
exact-repeat duration and sustained deviations. Outputs describe suspicious
behaviour for review; they do not diagnose hardware causes or repair observations.

The completed local run processed **2,836,536 development observations** and
replayed **133,920 fresh January 2025 observations** using frozen models and
thresholds. All 56 original columns matched the verified native inputs exactly.
The current suite includes observation integrity, causal streaming parity,
event/partition isolation, session validation and dashboard data contracts.

**Real hardware-fault accuracy remains unmeasured.** The fresh replay produced
170 candidate intervals, not 170 confirmed failures. The reviewed official
records do not establish fault labels for this 2023–2025 scope.

The sealed February synthetic benchmark is complete: learned-model row F1 is
**0.689 on held-out time** and **0.374 at held-out Goodwin Creek**. The latter
flags **49.65% of wholly unmodified background rows**, so reliable transfer to
new stations has not been demonstrated. The existing minute detector has lower
temporal F1 (0.456), slightly higher held-out-station F1 (0.387), and substantially
lower background candidate rates. See the [final evidence report](docs/SIH_FINAL_RESULTS_20261001.md)
for the separate results and limitations. These are software-scenario metrics,
not real hardware-fault accuracy.

- [Completed minute experiment and reproduction](docs/MINUTE_DETECTION_20260929.md)
- [Official event-evidence research](docs/research/surfrad_events_20260928/VERIFICATION.md)
- [Source verification](docs/research/surfrad_20260926/VERIFICATION.md)
- [Real-data rules](docs/REAL_DATA_RESEARCH.md)

## Run the operational dashboard

```powershell
pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
python scripts/serve_dashboard.py 8501
```

Open `http://127.0.0.1:8501`. The **Live detection** page sends original records
incrementally to a bounded-state backend. Scores are computed as rows arrive,
not read from precomputed results. It shows prediction warm-up, candidate
evidence, severity, inspection actions and data availability. This is historical
replay through a live inference engine, not a connected physical station.

Choose **Synthetic scenario stream** and click **Start synthetic stream** to
apply one of nine software faults after 180 unchanged minutes of real baseline
context. Baseline, synthetic input, deliberate changes and detector alerts have
separate markers. The same backend performs the actual scoring; the scenario
markers are not supplied as model predictors.

The **USA · Real observations** page inspects saved replay artifacts. The
**Detection benchmark** page keeps development and final-test evidence separate.
The API defaults to `artifacts_minute_20260928`; set `SKYGUARD_ARTIFACTS` to
`artifacts_minute_fresh_20260928` for January. Set `SKYGUARD_SIH_BENCHMARK` to a
new scenario-model artifact directory when comparing another frozen run.
The old Streamlit entry remains available as a legacy viewer.

The separate **India · Synthetic scenarios** page provides the user-requested
Indian demonstration based on archived Indian NOAA ISD reports. It compares
unchanged source values with explicitly synthetic scenarios. Calculated RH and
sea-level pressure keep their actual meanings. These scenarios are excluded
from the real-data training and accuracy claims. See the
[Indian station demo guide](docs/INDIAN_STATION_DEMO.md).

## Scenario training and independent evaluation

The following records the completed experiment's workflow. February 2025 has
been consumed once; do not rerun it as an untouched test or use its results for
model selection. Further model improvements require a newly declared holdout.
The completed workspace artifacts are in `artifacts_sih_final_20260930`.

```powershell
python scripts/run_sih_benchmark.py --stage develop --scenarios data/sih_scenarios_v2_reviewed_20260930
python scripts/select_sih_model.py --out artifacts_sih_tree_selected_20260930
python scripts/seal_sih_model.py --out artifacts_sih_final_20260930
python scripts/fetch_sih_holdout.py
python scripts/run_sih_benchmark.py --stage evaluate --out artifacts_sih_final_20260930 --scenarios data/sih_scenarios_v2_reviewed_20260930
```

Use new output/scenario directories for a new experiment. The source windows
are assigned to fit, selection, probability-calibration and final-test periods
before fault generation. All variants and their history stay in one partition.
Goodwin Creek is excluded from scenario-model development. A project-wide
consumption receipt prevents an examined final period becoming a new test just
by changing output directories. Predictions and synthetic values never overwrite
original fields. `no_injection` means no software intervention, not confirmed
normal hardware. Real weather can still be unusual.

Reports compare strict row metrics, event recall/delay, classifier confusion and
probability reliability against simple methods on the same source windows.
Repository research is not a head-to-head performance ranking:
[inspected implementations and evidence](docs/research/competitors_20260930/RESEARCH.md).

See [implementation/evidence plan](docs/SIH_COMPLETION_PLAN.md) and
[live API and measured parity](docs/LIVE_DETECTION_20260930.md).
Use the [deployment guide](docs/SIH_DEPLOYMENT.md) to build and verify a local
demo ZIP containing the frozen models, selected replay data and evidence.

## Verify

```powershell
python -m pytest tests/ -q
npm --prefix frontend run build
python docs/research/competitors_20260930/audit_receipts.py
```

The frontend build includes strict TypeScript checking. Tests use synthetic
fixtures only as software tests. Legacy deep-model tests additionally require
PyTorch; the active minute/scenario dashboard does not require a GPU.

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
