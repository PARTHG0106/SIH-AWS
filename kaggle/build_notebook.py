"""Generate the offline notebook with expected training-source fingerprints.

This generator does not prepare data or launch training. The generated notebook
refuses incompatible datasets before importing or fitting forecasting models.
Training executes source in the processed dataset, preserving the source bundle
-> online builder -> processed dataset -> training notebook release chain.
Regenerate after editing src/awsad; stale attached source is rejected.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_manifest(root: Path = ROOT) -> dict:
    """Fingerprint transitive training/preflight imports without bundling code.

    Walk imports inside functions too. The verified provider adapter, acquisition
    and training admission checks travel through the same pinned source chain.
    """
    pending = ["awsad", "awsad.real_observation_pipeline", "awsad.data.training_contract",
               "awsad.data.surfrad", "awsad.data.fetch_surfrad"]
    seen, files = set(), {}
    while pending:
        module = pending.pop()
        if module in seen or not (module == "awsad" or module.startswith("awsad.")):
            continue
        seen.add(module)
        stem = root / "src" / Path(*module.split("."))
        path = stem.with_suffix(".py") if stem.with_suffix(".py").is_file() else stem / "__init__.py"
        if not path.is_file():
            continue  # Namespace package or a symbol imported from a module.
        raw = path.read_bytes()
        files[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
        package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
        pending.extend(".".join(package[:i]) for i in range(1, len(package) + 1))
        for node in ast.walk(ast.parse(raw, filename=str(path))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names if alias.name.startswith("awsad"))
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                target = ".".join(prefix + (node.module.split(".") if node.module else []))
                pending.append(target)
                pending.extend(target + "." + alias.name for alias in node.names if alias.name != "*")
    for required in ("src/awsad/__init__.py", "src/awsad/real_observation_pipeline.py",
                     "src/awsad/data/training_contract.py", "src/awsad/data/surfrad.py",
                     "src/awsad/data/fetch_surfrad.py", "src/awsad/data/acquisition.py"):
        if required not in files:
            raise RuntimeError("missing required training source: " + required)
    files = dict(sorted(files.items()))
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "role": "expected_processed_training_source",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


BOOTSTRAP = r'''import hashlib, importlib, json, os, pathlib, shutil, sys, tempfile, zipfile
from datetime import datetime, timezone
sys.dont_write_bytecode = True

WORKDIR = pathlib.Path("/kaggle/working")
INPUTDIR = pathlib.Path("/kaggle/input")
WORKDIR.mkdir(parents=True, exist_ok=True)
PREFLIGHT = None  # Re-running bootstrap cannot reuse an earlier successful check.

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + CODE_MANIFEST["sha256"][:10]
OUT = WORKDIR / ("artifacts_" + RUN_ID)
OUT.mkdir()
(OUT / "expected_training_source.json").write_text(json.dumps(CODE_MANIFEST, indent=2))

def _decode_manifest(raw):
    if len(raw) > 2_000_000:
        raise RuntimeError("processed bundle manifest exceeds size limit")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise RuntimeError("processed bundle manifest must be an object")
    return manifest

def _read_manifest(root):
    path = root / "MANIFEST.json"
    if not path.is_file():
        return {}
    if path.stat().st_size > 2_000_000:
        raise RuntimeError("processed bundle manifest exceeds size limit")
    return _decode_manifest(path.read_bytes())

def _relative_path(rel):
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel:
        raise RuntimeError("unsafe dataset path: " + str(rel))
    path = pathlib.PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != rel or rel == ".":
        raise RuntimeError("unsafe dataset path: " + rel)
    return path

def locate_bundles():
    candidates = []
    for metadata in sorted(INPUTDIR.glob("**/data/processed/source_metadata.json")):
        root = metadata.parents[2]
        candidates.append({"kind": "root", "path": root, "prefix": "",
                           "manifest": _read_manifest(root)})
    for archive in sorted(INPUTDIR.glob("**/*.zip")):
        try:
            with zipfile.ZipFile(archive) as zf:
                names = set(zf.namelist())
                for name in sorted(names):
                    suffix = "data/processed/source_metadata.json"
                    if name.endswith(suffix):
                        prefix = name[:-len(suffix)]
                        manifest_name = prefix + "MANIFEST.json"
                        if prefix:
                            _relative_path(prefix.rstrip("/"))
                        if manifest_name in names and zf.getinfo(manifest_name).file_size > 2_000_000:
                            raise RuntimeError("processed bundle manifest exceeds size limit")
                        blob = _decode_manifest(zf.read(manifest_name)) if manifest_name in names else {}
                        candidates.append({"kind": "zip", "path": archive,
                                           "prefix": prefix, "manifest": blob})
        except zipfile.BadZipFile:
            continue
    return candidates

candidates = locate_bundles()
if not candidates:
    raise FileNotFoundError("Attach the verified SURFRAD bundle containing data/processed/source_metadata.json. "
                            "Rebuild the online source → builder → processed dataset release chain first.")
# An unpacked tree and its ZIP may describe the same immutable dataset build.
identities = {hashlib.sha256(json.dumps(c["manifest"], sort_keys=True).encode()).hexdigest()
              if c["manifest"] else str(c["path"]) for c in candidates}
if len(identities) != 1:
    raise RuntimeError("Ambiguous data inputs; attach exactly one processed build: " + repr(sorted(identities)))
chosen = sorted(candidates, key=lambda c: (c["kind"] != "root", str(c["path"])))[0]
if chosen["kind"] == "root":
    BUNDLE = chosen["path"]
else:
    BUNDLE = pathlib.Path(tempfile.mkdtemp(prefix="skyguard-training-input-"))
    extracted = set()
    with zipfile.ZipFile(chosen["path"]) as zf:
        for member in zf.infolist():
            if not member.filename.startswith(chosen["prefix"]):
                continue
            rel = member.filename[len(chosen["prefix"]):].rstrip("/")
            if not rel:
                continue
            parts = _relative_path(rel)
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError("dataset archives must not contain symbolic links")
            destination = BUNDLE.joinpath(*parts.parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                if rel in extracted:
                    raise RuntimeError("duplicate dataset archive member: " + rel)
                extracted.add(rel)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
BUNDLE_MANIFEST = chosen["manifest"]
PROCESSED = BUNDLE / "data" / "processed"
print("selected data:", BUNDLE)

# Execute the code copied through the user's source -> builder -> processed
# release chain. A newer notebook must not silently train with older models.
source_mismatches = []
bundle_root = BUNDLE.resolve()
for rel, expected in CODE_MANIFEST["files"].items():
    path = BUNDLE.joinpath(*_relative_path(rel).parts)
    safe = path.is_file() and not path.is_symlink() and bundle_root in path.resolve().parents
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if safe else None
    if actual != expected:
        source_mismatches.append({"path": rel, "expected": expected, "actual": actual})
source_report = {"eligible": not source_mismatches,
                 "expected_source_sha256": CODE_MANIFEST["sha256"],
                 "checked_files": len(CODE_MANIFEST["files"]), "mismatches": source_mismatches}
(OUT / "source_preflight.json").write_text(json.dumps(source_report, indent=2))
if source_mismatches:
    print(json.dumps(source_report, indent=2))
    raise RuntimeError("Processed bundle contains stale or missing training source. "
                       "Rebuild skyguard_source_bundle, upload skyguard-sih26073, run the online builder, "
                       "publish its output as skyguard-sih26073-processed, then rerun this notebook.")
SOURCE_ROOT = BUNDLE
for module_name in list(sys.modules):
    if module_name == "awsad" or module_name.startswith("awsad."):
        del sys.modules[module_name]
sys.path.insert(0, str(BUNDLE / "src"))
importlib.invalidate_caches()
import awsad
assert pathlib.Path(awsad.__file__).resolve() == (BUNDLE / "src/awsad/__init__.py").resolve()
for rel in CODE_MANIFEST["files"]:
    destination = OUT / "training_source" / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLE / rel, destination)
(OUT / "training_source" / "CODE_MANIFEST.json").write_text(json.dumps(CODE_MANIFEST, indent=2))
print("verified processed-bundle training source SHA-256:", CODE_MANIFEST["sha256"])

# Fail before importing models; verification itself reads archived observations.
# A fresh build ID or a hash is integrity evidence, not proof of real readings.
from awsad.data.training_contract import require_eligible_training_data, TrainingDataContractError
try:
    DATA_PREFLIGHT = require_eligible_training_data(PROCESSED, report_path=OUT / "data_preflight.json")
except TrainingDataContractError as exc:
    print(json.dumps(exc.report, indent=2))
    print("Training did not start. Preflight report:", OUT / "data_preflight.json")
    raise

if (BUNDLE_MANIFEST.get("schema_version") != 1
        or BUNDLE_MANIFEST.get("bundle_role") != "processed_training_bundle"
        or BUNDLE_MANIFEST.get("training_data_policy") != "real_observations_only"
        or BUNDLE_MANIFEST.get("training_source_sha256") != CODE_MANIFEST["sha256"]):
    raise RuntimeError("Expected the matching verified processed training release")
source_provenance = BUNDLE / "provenance" / "source_manifest.json"
if (not source_provenance.is_file()
        or hashlib.sha256(source_provenance.read_bytes()).hexdigest() != BUNDLE_MANIFEST.get("source_manifest_sha256")):
    raise RuntimeError("Processed release does not preserve its exact source-release manifest")
hashes = BUNDLE_MANIFEST.get("sha256")
if not isinstance(hashes, dict) or not hashes:
    raise RuntimeError("input bundle requires a complete SHA-256 manifest")
bundle_root = BUNDLE.resolve()
for rel, expected in hashes.items():
    parts = _relative_path(rel)
    expected = str(expected).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise RuntimeError("invalid input SHA-256: " + str(rel))
    path = bundle_root.joinpath(*parts.parts).resolve()
    if bundle_root not in path.parents or not path.is_file():
        raise RuntimeError("missing or unsafe input file: " + str(rel))
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError("input bundle hash mismatch: " + str(rel))
paths = list(BUNDLE.rglob("*"))
if any(path.is_symlink() for path in paths):
    raise RuntimeError("processed bundle must not contain symbolic links")
actual_files = {p.relative_to(BUNDLE).as_posix() for p in paths
                if p.is_file() and p != BUNDLE / "MANIFEST.json"}
if actual_files != set(hashes):
    raise RuntimeError("input manifest does not cover exactly the retained bundle files")
BUILD_ID = str(BUNDLE_MANIFEST.get("build_id", "unknown"))
PREFLIGHT = DATA_PREFLIGHT  # Publish eligibility only after every release check succeeds.
print("verified data build:", BUILD_ID, "| files:", len(hashes))
'''

CONFIG = r'''import os, platform, time
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
if not PREFLIGHT or not PREFLIGHT.get("eligible"):
    raise RuntimeError("Data/source preflight must succeed before importing models")
import numpy as np
import pandas as pd
from awsad.real_observation_pipeline import RealObservationConfig

# CPU budgets affect runtime; model selection still compares against observed
# persistence and seasonal baselines on validation only.
PROFILE = os.environ.get("SKYGUARD_PROFILE", "standard")
PROFILES = {
    "fast": {"max_iter": 40, "max_train_rows": 8000, "max_validation_rows": 4000},
    "standard": {"max_iter": 80, "max_train_rows": 20000, "max_validation_rows": 10000},
    "full": {"max_iter": 160, "max_train_rows": 60000, "max_validation_rows": 20000},
}
if PROFILE not in PROFILES:
    raise ValueError("SKYGUARD_PROFILE must be fast, standard, or full")
CONFIG = RealObservationConfig(
    **PROFILES[PROFILE], horizon_minutes=(60,), thinning_minutes=60,
    train_end="2024-09-01T00:00:00Z", validation_end="2024-11-01T00:00:00Z",
    test_end="2025-01-01T00:00:00Z", holdout_station_ids=("gwn",),
    random_seed=42, max_cpu_threads=4, evaluate_test=True,
)
CONFIG.validate()
RUN_CONFIG = {"profile": PROFILE, "device": "cpu", "config": asdict(CONFIG),
              "dataset_build_id": BUILD_ID, "training_source_sha256": CODE_MANIFEST["sha256"]}
versions = {"python": platform.python_version()}
for distribution in ("numpy", "pandas", "scikit-learn", "pyarrow"):
    try:
        versions[distribution] = version(distribution)
    except PackageNotFoundError:
        versions[distribution] = None
RUN_CONFIG["runtime_versions"] = versions
(OUT / "run_config.json").write_text(json.dumps(RUN_CONFIG, indent=2))
print(json.dumps(RUN_CONFIG, indent=2))
'''

DATA_SANITY = r'''if not PREFLIGHT or not PREFLIGHT.get("eligible"):
    raise RuntimeError("Verified data preflight must succeed before reading observations")
observations = pd.read_parquet(PROCESSED / "observations.parquet")
source_metadata = json.loads((PROCESSED / "source_metadata.json").read_text())
required = {"observation_id", "timestamp", "station_id", "source",
            "temperature_c", "pressure_hpa", "relative_humidity_pct"}
if not required <= set(observations):
    raise RuntimeError("Verified observations are missing model inputs: " + repr(sorted(required - set(observations))))
if observations.empty or observations["observation_id"].duplicated().any():
    raise RuntimeError("Observations must be nonempty and have unique immutable record IDs")
if "label" in observations and observations["label"].notna().any():
    raise RuntimeError("This experiment has no confirmed hardware labels; unexpected label values are rejected")
coverage = {"rows": len(observations), "station_ids": sorted(observations.station_id.astype(str).unique()),
            "sources": sorted(observations.source.astype(str).unique()),
            "first_timestamp": str(observations.timestamp.min()), "last_timestamp": str(observations.timestamp.max()),
            "available_values": {c: int(observations[c].notna().sum()) for c in
                                 ("temperature_c", "pressure_hpa", "relative_humidity_pct")},
            "fault_labels": "unknown; not manufactured or inferred from provider QC"}
(OUT / "coverage.json").write_text(json.dumps(coverage, indent=2))
print(json.dumps(coverage, indent=2))
'''

TRAIN = r'''if not PREFLIGHT or not PREFLIGHT.get("eligible"):
    raise RuntimeError("Data preflight must succeed before training")
from awsad.real_observation_pipeline import run_real_observation_pipeline
t0 = time.perf_counter()
report = run_real_observation_pipeline(observations, OUT, CONFIG, source_metadata=source_metadata)
elapsed = time.perf_counter() - t0
training_manifest = {**RUN_CONFIG, "wall_clock_seconds": elapsed,
                     "dataset_manifest": BUNDLE_MANIFEST, "data_preflight": PREFLIGHT,
                     "code_manifest": CODE_MANIFEST, "source_metadata": source_metadata}
(OUT / "training_manifest.json").write_text(json.dumps(training_manifest, indent=2, default=str))
print("Training status:", report["status"], "| elapsed minutes:", round(elapsed / 60, 2))
if report["status"] != "completed":
    raise RuntimeError("Some required channel models could not be fitted; inspect metrics.json")
'''

METRICS = r'''rows = []
for horizon, horizon_report in report.get("horizons", {}).items():
    for channel, result in horizon_report.get("channels", {}).items():
        selected = result.get("selected_model")
        for partition, evaluation in result.get("evaluation", {}).get("test", {}).items():
            if not isinstance(evaluation, dict):
                continue
            for candidate, metrics in evaluation.get("provider_accepted_forecasting", {}).items():
                rows.append({"horizon": horizon, "channel": channel, "station_partition": partition,
                             "candidate": candidate, "selected_on_validation": candidate == selected,
                             "mae": metrics.get("mae"), "rmse": metrics.get("rmse"),
                             "macro_group_mae": metrics.get("macro_group_mae"), "rows": metrics.get("n")})
if rows:
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "forecast_comparison.csv", index=False)
    print(comparison.to_string(index=False))
print("Protocol:", json.dumps(report.get("protocol", {}), indent=2))
print("These are forecasting errors against retained provider observations. "
      "They are not hardware-fault precision, recall, classification accuracy, or false-positive rates. "
      "Provider-QC agreement, when available, is reported separately in metrics.json.")
'''

PLOTS = r'''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
scored = pd.read_parquet(OUT / "scored_observations.parquet")
test_rows = scored.loc[scored.evaluation_split == "test"]
if not test_rows.empty:
    station = sorted(test_rows.station_id.astype(str).unique())[0]
    replay = test_rows.loc[test_rows.station_id.astype(str) == station].sort_values("timestamp").head(168)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].scatter(replay.timestamp, replay.temperature_c, label="Provider-observed temperature", s=8)
    axes[0].scatter(replay.timestamp, replay["60m__temperature_c__prediction"],
                    label="One-hour model forecast (not an observation)", s=8)
    axes[0].set_ylabel("deg C")
    axes[0].legend()
    axes[0].set_title("Historical observation replay: " + station)
    axes[1].scatter(replay.timestamp, replay["60m__anomaly_score"], label="Derived residual score", s=8)
    axes[1].set_ylabel("Residual score")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "observed_replay.png", dpi=120)
    plt.close(fig)
    replay.to_csv(OUT / "observed_replay.csv", index=False)
    print("Saved historical replay, preserving observed values and unavailable-score gaps.")
'''

EXPORT = r'''archive = WORKDIR / ("skyguard_real_artifacts_" + RUN_ID + ".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(OUT))
print("Artifacts:", OUT)
print("Download:", archive, f"({archive.stat().st_size / 1e6:.1f} MB)")
'''


def build_notebook(root: Path = ROOT) -> dict:
    manifest = source_manifest(root)
    specs = [
        ("markdown", """# SkyGuard AI — real-observation forecasting and anomaly scoring

This CPU experiment uses provider-verified SURFRAD temperature, independently measured
relative humidity and station pressure. The admission check verifies original archived
records and field lineage before the observation table is loaded. Missing measurements
stay missing, and unknown hardware-fault labels stay unknown.

The release chain remains: `skyguard_source_bundle` → `skyguard-sih26073` → online
`aws_data_builder.ipynb` → `skyguard-sih26073-processed` → this notebook. Training
runs `src/awsad` from that processed dataset. Expected training-source hashes are
generated locally; stale attached code is refused instead of silently ignored.
Regenerate with `python kaggle/build_notebook.py` after editing source and before
uploading. Accepted source files, hashes, data preflight and resolved configuration
are exported. Legacy injected/reanalysis bundles are rejected.

The training period ends on 1 September 2024, validation ends on 1 November 2024,
and November–December 2024 is held back for the final chronological test. Goodwin
Creek (`gwn`) is held out from model fitting and validation selection. Forecasts
are compared with persistence, an observed seasonal baseline and robust changes.
The model is selected on validation only. Results measure forecasting, not confirmed
hardware-fault detection; this US network does not establish performance on Indian AWS.
"""),
        ("markdown", "## 1 · Processed-bundle source and data eligibility"),
        ("code", "# Expected source fingerprints; regenerate after editing local training code.\n"
         + "CODE_MANIFEST = " + repr(manifest) + "\n" + BOOTSTRAP),
        ("markdown", "## 2 · Run budget\n\nSet `SKYGUARD_PROFILE` to `fast`, `standard` (default), or `full` before execution. "
         "All profiles use CPU models and the same validation protocol; larger budgets do not guarantee improvement."),
        ("code", CONFIG),
        ("markdown", "## 3 · Coverage and chronological separation\n\nMissing observations and unknown labels remain explicit. "
         "The fixed experiment configuration declares the station holdout before training."),
        ("code", DATA_SANITY),
        ("markdown", "## 4 · Train and evaluate\n\nOnly data accepted by the preflight may reach this cell. "
         "Training/model selection must use training and validation evidence; the test split is reserved for the final report."),
        ("code", TRAIN),
        ("markdown", "## 5 · Forecast comparison on held-back observations\n\nReport MAE/RMSE, station holdouts, score availability and provider-QC agreement separately. "
         "Unknown fault labels cannot support hardware-fault accuracy claims."),
        ("code", METRICS),
        ("markdown", "## 6 · Historical observed replay and separate model predictions"),
        ("code", PLOTS),
        ("markdown", "## 7 · Export artifacts and exact training source"),
        ("code", EXPORT),
    ]
    cells = []
    for index, (kind, source) in enumerate(specs):
        cell = {"cell_type": kind, "id": f"skyguard-{index:02d}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            compile(source, f"cell_{index}", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "skyguard": {"training_source_sha256": manifest["sha256"],
                         "data_policy": "verified_original_SURFRAD_observations; unknown_fault_labels"},
            "kaggle": {"accelerator": "none", "dataSources": [], "dockerImageVersionId": None,
                       "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    out = ROOT / "kaggle" / "aws_anomaly_training.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Notebook written: {out} ({out.stat().st_size:,} bytes)")
    print("Training source SHA-256:", notebook["metadata"]["skyguard"]["training_source_sha256"])
    print("No training was launched. Incompatible datasets fail the notebook preflight.")


if __name__ == "__main__":
    main()
