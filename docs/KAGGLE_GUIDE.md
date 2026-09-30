# Kaggle: real observations, online build, offline training

The current path uses original NOAA SURFRAD temperature, independently measured
RH and station pressure. It never injects faults, fills missing observations or
labels unreviewed readings normal. The old ISD/injected processed versions are
rejected. See [provider verification](research/surfrad_20260926/VERIFICATION.md).

| Stage | Kaggle item | Internet | Accelerator |
|---|---|---|---|
| Source | krishnagupta02468/skyguard-sih26073 | — | — |
| Builder | skyguard-ai-data-builder-sih26073 | On | CPU |
| Processed observations | krishnagupta02468/skyguard-sih26073-processed | — | — |
| Training | skyguard-ai-aws-anomaly-training-sih26073 | Off | CPU |

Verified on 26 September 2026: source dataset **v19** → builder **v18** →
processed dataset **v9** → training **v14**, completed using only the named
processed dataset attachment. The full provenance checks passed, and models
and scored observations match the previous builder-output run exactly.
See [the completed experiment](REAL_TRAINING_20260926.md).

## Native-minute continuation (29 September 2026)

The new `kaggle/aws_minute_detection.ipynb` uses the original daily files already
retained in processed v9. It embeds/verifies updated minute source separately
from that release's archived hourly code. It reconstructs native observations
through October 2024, fits/selects/calibrates on separate dates and exports a
real replay with unknown event-review proposals. Use CPU, with Internet off.

Generate with `python kaggle/build_minute_notebook.py`; `kaggle_minute/` contains
matching upload metadata and notebook. Attach
`krishnagupta02468/skyguard-sih26073-processed`. This notebook was validated
locally, not published or run on Kaggle. Existing hourly v14 is unchanged.
See [minute results and reproduction](MINUTE_DETECTION_20260929.md). The remaining
guide describes the existing source-builder-hourly workflow.

## Refresh source and notebooks

After editing source, regenerate both notebooks and stage the source release:

~~~powershell
python kaggle/build_notebook.py
python kaggle/build_data_builder.py
python scripts/prepare_kaggle_upload.py
~~~

Upload data/kaggle_staging/skyguard-sih26073.zip as a new source dataset version.
Its manifest must declare online_builder_source, real_observations_only,
processed_data_included: false, and parser_version: surfrad-observed-v1.
It contains source, configuration and documentation, without credentials,
model artifacts or training observations.

## Build original observations

Run kaggle/aws_data_builder.ipynb (identical copy in kaggle_builder/) with
the new source dataset attached. It verifies source hashes, then acquires the
provider-listed daily files for Bondville, Fort Peck and Goodwin Creek in
2023–2024. Four requests may run concurrently; failed downloads stop admission.
Missing days in the provider inventory remain missing.

The modeling view selects existing minute00 records from published one-minute
averages. It does not construct hourly observations. The complete original
daily files, source documents, raw values, QC codes and field lineage travel
with the processed release. The independent verifier reconstructs every
retained row and checks the whole table, including missing and unknown states.

Output: /kaggle/working/awsad_real_bundle_<build-id>/. Publish the complete
directory, or a ZIP with its MANIFEST.json at the root, as a new version of
skyguard-sih26073-processed. It includes src/, provenance/ and data/processed/;
uploading only the Parquet loses required evidence.

For a local build using the identical verification/build/package notebook
cells, first acquire the originals, then use the cached archive:

~~~powershell
python scripts/fetch_real_surfrad.py
python kaggle/build_data_builder.py --source-bundle data/kaggle_staging/skyguard_source_bundle --archive-dir data/raw/surfrad_original --output-root data/real_releases
~~~

This local option avoids downloading the same immutable source files again.
It produces the same portable processed layout and source-provenance chain.

## Train and evaluate

Attach the new processed dataset to kaggle/aws_anomaly_training.ipynb and run
all cells. The notebook verifies the expected code version, release hashes,
provider evidence and every processed observation before fitting models.
It needs CPU only; no GPU time is required.

The default experiment forecasts one hour ahead using past measured readings,
their differences and explicit time features. Candidates include persistence,
the previous day's observation, robust change, gradient boosting of levels,
and gradient boosting of changes with full or half-strength predictions.
The first validation segment chooses the model by average station error.
Reserved training and validation segments determine station-specific residual
scales and alert thresholds; unseen stations use the pooled fallback.

Splits are fixed before evaluation:

- Train: 2023 through August 2024, Bondville and Fort Peck.
- Validation: September–October 2024, separately divided for model selection
  and threshold calibration.
- Test: November–December 2024. Goodwin Creek is a station holdout and never
  contributes to training, model selection or threshold calibration.

To compare configurations locally before opening the test period:

~~~powershell
python scripts/run_real_observation_train.py --dataset-dir <bundle>/data/processed --out artifacts_real_selection --config configs/real_surfrad.json --selection-only
~~~

After freezing the configuration, use a new output directory and
--evaluate-test. Keep artifacts from each experiment; do not overwrite or
tune on test results. The notebook's standard profile is the 80-iteration,
20,000-training-row CPU budget. Larger profiles require fresh validation before
being treated as an improvement.

Outputs include metrics.json, actual observations with separate predictions
and scores, model artifacts, exact training record IDs and source fingerprints.
These measure forecasting error, scored coverage and alert rate. Provider-QC
agreement is a separate result. None is confirmed hardware-fault accuracy,
and US SURFRAD validation is not validation on Indian IMD AWS.

The old dashboard/model-export format remains a separate legacy path; these
forecast artifacts must not be passed to its old nine-class fault model loader.
