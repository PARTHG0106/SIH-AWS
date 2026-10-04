# SIH submission packaging and local deployment

The packager creates a local ZIP from an explicit list of source files, final
models, verification evidence and selected data. It does not publish anything.
The launcher serves the React dashboard and incremental API on `127.0.0.1:8501`.
This is a local demonstration package, not a connected station installation.

## Prepare and inspect

The synthetic scenario model's frozen February evaluation is complete in
`artifacts_sih_final_20260930`. Its station-transfer limitation remains part of
the release evidence; see [final results](SIH_FINAL_RESULTS_20261001.md). Rebuild
the frontend and check this completed directory before packaging:

```powershell
npm --prefix frontend run build
.venv/Scripts/python.exe scripts/package_sih_release.py --check --pattern-dir artifacts_sih_final_20260930 --include-data --include-india --include-originals
```

`--check` writes no archive. It reports missing files, unfinished final metrics,
source/model hash changes, incompatible model dependency versions, and a stale
or incomplete frontend build. The final result must also match its completed
holdout-consumption receipt. A package cannot be created while those errors
remain. The original README and source documentation are copied as they stand;
packaging does not rewrite claims or regenerate observations.

`--pattern-dir` selects one explicit workspace-relative artifact directory. The
default is `artifacts_sih_20260930`; for this workspace, pass the completed
`artifacts_sih_final_20260930` explicitly. The earlier tree-selected and release
directories do not contain this completed final evaluation. Absolute paths,
parent traversal, hidden directories, symlinks
and junctions are rejected. Only the listed model/metrics files are read from that
directory. The archive always stores them under `artifacts_sih_20260930`, so its
launcher path stays stable; the inventory records the original selected paths.
The recorded February test is consumed. Packaging reads its existing results;
do not rerun or retune the evaluation as a packaging step.

The source seal covers the exact `awsad` package, `run_sih_benchmark.py`, and
`run_minute_detection.py`. When the freeze contains `selection_script_sha256`,
the separate `select_sih_model.py` seal is verified too. Selection code is included
without changing any frozen manifest or its attribution.

The recommended local demonstration package includes the three January 2025
SURFRAD replay shards. January has already been examined and is not an untouched
evaluation holdout. Optional Indian baselines stay on their separately labelled
synthetic demonstration page.

## Create a release only after checks pass

```powershell
.venv/Scripts/python.exe scripts/package_sih_release.py --pattern-dir artifacts_sih_final_20260930 --include-data --include-india --include-originals
```

The default destination is `out_sih_release/`. The ZIP filename contains the
inventory hash. Existing ZIPs and hash receipts are never overwritten. Use
`--out NEW_DIRECTORY` to retain a separate build. Identical inputs produce the
same ZIP bytes through sorted paths, fixed ZIP timestamps and stable metadata.
The adjacent `.zip.sha256` records the final ZIP hash; the contained
`release_manifest.json` records the size, SHA-256, source path and category of
every other file. A hash inventory detects changes; it is not a digital signature.

Selection flags are independent and explicit:

| Flag | Included data |
|---|---|
| none | Source, built frontend, frozen models, metrics and provenance; replay measurements are omitted |
| `--include-data` | Exactly `bon`, `fpk`, `gwn` January 2025 scored replay shards, with original fields unchanged |
| `--include-india` | Exactly six reviewed 2024 Indian baseline files and their catalog/manifest |
| `--include-originals` | Requires `--include-data`; includes only January acquisition-manifest-listed originals, inventories and provider documents; also the six Indian source CSVs and station metadata when India is selected |

No complete historical corpus, training scenario arrays, legacy injected data,
credentials, `.env` files, logs, Git state, environment folders, or arbitrary
workspace content are selected. Frozen source snapshots are retained exactly.
Some legacy Python modules are present because the frozen protocol seals the
entire package file set; legacy dataset builder scripts are not included or run.

## Run the extracted package

Use Python matching `release_settings.json` (the tested environment is Python
3.14). From the extracted directory:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-runtime.txt
.venv/Scripts/python.exe launch_sih.py --check
.venv/Scripts/python.exe launch_sih.py
```

On Linux/macOS the interpreter path is `.venv/bin/python`. Dependency installation
needs an available package index or a separately prepared wheelhouse; dependencies
are not bundled. Node is unnecessary to serve the prebuilt frontend. To rebuild
it, run `npm ci --prefix frontend` followed by `npm --prefix frontend run build`.
Rebuilding changes the checked release and requires a new package/inventory.

The launcher verifies every packaged file, the recorded Python version, and
the model's frozen dependency versions before loading the models. It sets:

- `SKYGUARD_ARTIFACTS` to the packaged fresh January replay directory.
- `SKYGUARD_SIH_BENCHMARK` to the final synthetic-pattern artifact directory.
- `SKYGUARD_INDIAN_DEMO` to the separate Indian demo directory.

Use `--port 8600` to choose another port. A package built without `--include-data`
requires `--replay-dir PATH_TO_VERIFIED_JANUARY_BUNDLE`; otherwise the launcher
stops with an explicit missing-data message. `--india-dir` can point to an
external reviewed Indian demo bundle. If neither an included nor external Indian
bundle is available, that optional page cannot provide observations.

## Scope and attribution

The interface replays original observations through a stateful inference API and
can stream a labelled scenario copy from the dashboard. Applied changes, original
baseline values and detector candidates remain separate. Neither mode is a
connected physical AWS feed.
The final pattern model was calibrated/evaluated on separate software scenarios;
its probability is not real hardware-failure probability. The held-out station
produced learned candidates on 49.6451% of untouched-background rows; the package
does not resolve that generalization failure. Edge watchdog policy
and local host timing are separate evidence, not an ESP32 power or timing test.
The package provides no hardware drivers, authentication, durable queue,
multi-process session store, deployment supervision or station radio transport.

`THIRD_PARTY_ATTRIBUTION.md`, the frontend dependency licenses and
`docs/LICENSES.md` distinguish project rights, SURFRAD CC0 data, optional NOAA
ISD baselines, separately installed Python libraries and inspected public
research. The complete ZIP is not relicensed CC0. Packaged public research
evidence is `docs/research/competitors_20260930/source_index.json`: URLs,
retrieval metadata and response hashes only. Inspected third-party source bodies
and the full workspace `source_receipts.json` are not redistributed. No peer
implementation is claimed as project code. Raw acquisition metadata and original
hashes remain intact.

## Packaging verification

`python -m pytest tests/test_sih_package.py -q` uses only fake fixture files. It
checks token/environment exclusion, deterministic output, per-file inventories,
missing/changed model and build rejection, traversal/symlink rejection, refusal
to overwrite an existing release, and launcher integrity verification. No test
fixture is included in a claimed observation or evaluation dataset.

On 1 October the packager tests passed (31 passed, 1 skipped). An extracted
315-entry verification build passed its launcher inventory/runtime check and
27 live HTTP checks, including original/synthetic provenance and model warm-up.
The final archive is produced after these evidence notes are updated; its
adjacent `.zip.sha256` and `verification.json` identify the exact bytes and
confirm runtime identity with the HTTP-tested build. Full details are in
`docs/SIH_FINAL_RESULTS_20261001.md`. These are local software checks, not a
physical deployment validation.
