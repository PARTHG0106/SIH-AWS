# Training revision — 2026-09-25

The training code and notebook were revised and tested. **No new model accuracy
has been measured and no GPU training or online data-builder run was launched.**
The existing Kaggle processed dataset is incompatible with the binding real-data
requirement; the new entrypoints stop before expensive loading/training.

## Verified workflow and current versions

The intended release chain is unchanged:

1. `scripts/prepare_kaggle_upload.py` creates
   `data/kaggle_staging/skyguard_source_bundle/` and `skyguard-sih26073.zip`.
2. That source is uploaded as `krishnagupta02468/skyguard-sih26073`.
3. The online `kaggle/aws_data_builder.ipynb` produces a processed bundle.
4. That output is uploaded as `krishnagupta02468/skyguard-sih26073-processed`.
5. `kaggle/aws_anomaly_training.ipynb` runs the source included in that processed
   dataset. It checks that the training source matches the notebook's expected
   hashes; updating a notebook cannot silently run stale bundled code.

The live metadata audit verified source **v17** and processed **v8**, build
`20260925T105454Z-0e7c7c1e3699`, matching the newer supplied builder log. The
previous training log used `20260924T163929Z-2b075b1ebeac`. Local September 19
Parquets are not the authority for the latest online build.

See [the pinned remote metadata audit](research/kaggle_processed_20260925/README.md).
Only four small JSON files were downloaded, with hashes checked against the
processed manifest. The large observation arrays were not downloaded. Metadata
reports 96.87% synthetic validation positives, 96.32% synthetic test positives,
36.7–39.2% retained imputation by channel/split, derived NOAA RH, and sea-level
pressure. Remaining provider QC positives are not confirmed hardware faults.

## Changes

- **Causal scores:** QC persistence alerts begin only after enough observations
  have arrived. Fixed a reverse-run counting bug and removed future-data standard
  deviations/backfills. Autoencoders assign an error only to the last timestep
  of a trailing window. Forecaster warm-up no longer copies future errors back.
  Warm-up/no-evidence scores are model outputs, not normal ground-truth labels.
- **Separate stacker calibration:** each fit group's validation history is split
  chronologically into 60% fitting, 20% iteration selection, and 20% threshold
  calibration, with 168-hour elapsed-time gaps. Known positive runs shared by
  partitions are excluded. Model selection averages PR-AUC across eligible tuning
  groups, uses balanced station/class training weights and bounded row budgets.
  Holdout stations are excluded. These partitions are per-group, not synchronized
  across the whole network. The exported scorer and reported stacker now use the
  same threshold-only decision policy. Class-balanced classifier scores are not
  claimed to be calibrated fault probabilities.
- **Thresholds:** exact strict-F1 selection considers every tied score block and
  an explicit no-alarm option, instead of assuming anomalies occupy the top 10%.
  Unknown labels/nonfinite scores are excluded. No-alarm thresholds remain above
  constant scores when serving float32/float16 arrays. Sparse local supervision
  falls back to pooled calibration; absent supervision uses train-only POT.
- **Evaluation:** group separators affect event boundaries only, never AUC or
  prevalence. Unknown label gaps remain unknown. Reports include per-group F1
  and a deterministic group-bootstrap interval. Overlap/event computations use
  prefix counts and indexed timestamps instead of repeated corpus scans.
- **Runtime:** training indexes station/source rows once. Notebook coverage uses
  grouped scans. `fast`, `standard`, and `full` budgets are explicit; standard
  omits the costly 168-step autoencoder. This is a resource choice, not evidence
  of improved accuracy. No detector is forced to have a positive ensemble weight.
- **Admission:** local and notebook training require source/lineage evidence.
  Known legacy data are rejected before Parquet/GPU work. No provider-specific
  admission adapter exists yet, so changing a manifest cannot self-certify data.
  This is an explicit unfinished migration, not a ready real-data pipeline.

The old run's zero deep weights do not prove that larger networks help: its
validation search preferred the QC/spatial combination. The old weighted
ensemble point-F1 was about 0.516 on constructed labels. That number is retained
as historical context only, not a real-fault baseline or a new result.

## Validation and remaining work

The full offline test suite passed **99 tests plus 6 subtests**. Tests cover
causal prefix invariance, threshold search against brute force, float32 boundary
behavior, station/time isolation, unknown labels, event endpoint alignment,
stale-code rejection and preflight refusal before heavy imports. Synthetic
fixtures remain confined to tests and are not training/evaluation data.

The notebook is regenerated from the final source and its code cells/schema
validated. The refreshed source ZIP is a local package only; it has not been
published. Existing model checkpoints and thresholds require retraining after
these scoring changes. Do not mix them with the new scoring implementation.

To reach a compliant training run, the online builder must first be migrated
to original measured T/station-pressure/RH, with source-specific verification,
missingness and per-field lineage preserved. Independent reviewed labels are
needed for hardware-fault accuracy; absent labels require explicitly unlabelled
training and no fault-accuracy claim. Follow [the acceptance policy](REAL_DATA_RESEARCH.md).

Reproduce the software checks with the working Python environment and
`python -m pytest tests/ -q`. The local `.venv/Scripts/python.exe` is a broken
one-byte stub on this machine; validation used installed Python 3.14 with
`.venv/Lib/site-packages` and `src` on `PYTHONPATH`.
