"""Generate the online CPU builder for original, provider-verified observations.

By default this only generates notebooks. The explicit local-build entry point
uses the same verification and packaging cells with an existing raw archive.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_manifest(root: Path = ROOT) -> dict:
    spec = importlib.util.spec_from_file_location("skyguard_training_generator", root / "kaggle/build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_manifest(root)


BOOTSTRAP = r'''import hashlib, importlib, json, pathlib, shutil, sys, tempfile, zipfile
from datetime import datetime, timezone
sys.dont_write_bytecode = True
WORKDIR = pathlib.Path("/kaggle/working")
INPUTDIR = pathlib.Path("/kaggle/input")
WORKDIR.mkdir(parents=True, exist_ok=True)

def _source_manifest(raw):
    if len(raw) > 2_000_000:
        raise RuntimeError("source manifest exceeds size limit")
    blob = json.loads(raw)
    return blob if isinstance(blob, dict) and blob.get("bundle_role") == "online_builder_source" else None

def _relative_path(rel):
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel:
        raise RuntimeError("unsafe source path: " + str(rel))
    path = pathlib.PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != rel or rel == ".":
        raise RuntimeError("unsafe source path: " + rel)
    return path

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _verify_files(root, hashes):
    if not isinstance(hashes, dict) or not hashes or "MANIFEST.json" in hashes:
        raise RuntimeError("source requires a complete SHA-256 map excluding MANIFEST.json")
    resolved_root = root.resolve()
    for rel, expected in hashes.items():
        parts = _relative_path(rel)
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise RuntimeError("invalid source SHA-256: " + rel)
        path = root.joinpath(*parts.parts)
        if path.is_symlink() or resolved_root not in path.resolve().parents or not path.is_file():
            raise RuntimeError("missing or unsafe source file: " + rel)
        if _sha256(path) != expected:
            raise RuntimeError("source bundle hash mismatch: " + rel)
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("source bundle must not contain symbolic links")
    actual = {path.relative_to(root).as_posix() for path in paths
              if path.is_file() and path != root / "MANIFEST.json"}
    if actual != set(hashes):
        raise RuntimeError("source manifest must cover exactly the retained source files")

candidates = []
for path in sorted(INPUTDIR.glob("**/MANIFEST.json")):
    if not (path.parent / "src/awsad/__init__.py").is_file():
        continue  # Archived documentation manifests are not attached source roots.
    if path.stat().st_size > 2_000_000:
        continue
    raw = path.read_bytes()
    try:
        manifest = _source_manifest(raw)
    except (ValueError, UnicodeError):
        continue
    if manifest:
        candidates.append({"kind": "root", "path": path.parent, "prefix": "", "raw": raw, "manifest": manifest})
for path in sorted(INPUTDIR.glob("**/*.zip")):
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for member in zf.infolist():
                if not member.filename.endswith("MANIFEST.json") or member.file_size > 2_000_000:
                    continue
                prefix = member.filename[:-len("MANIFEST.json")]
                if prefix + "src/awsad/__init__.py" not in names:
                    continue  # An archived manifest without adjacent source is evidence only.
                raw = zf.read(member)
                try:
                    manifest = _source_manifest(raw)
                except (ValueError, UnicodeError):
                    continue
                if manifest:
                    if prefix:
                        _relative_path(prefix.rstrip("/"))
                    candidates.append({"kind": "zip", "path": path, "prefix": prefix, "raw": raw, "manifest": manifest})
    except zipfile.BadZipFile:
        continue
if not candidates:
    raise FileNotFoundError("Attach the current skyguard-sih26073 source-only dataset containing MANIFEST.json")
identities = {hashlib.sha256(candidate["raw"]).hexdigest() for candidate in candidates}
if len(identities) != 1:
    raise RuntimeError("Ambiguous source inputs; attach exactly one source release")
chosen = sorted(candidates, key=lambda candidate: (candidate["kind"] != "root", str(candidate["path"])))[0]
SOURCE_MANIFEST = chosen["manifest"]
SOURCE_MANIFEST_BYTES = chosen["raw"]
SOURCE_MANIFEST_SHA256 = hashlib.sha256(SOURCE_MANIFEST_BYTES).hexdigest()
if (SOURCE_MANIFEST.get("schema_version") != 1
        or SOURCE_MANIFEST.get("processed_data_included") is not False
        or SOURCE_MANIFEST.get("training_data_policy") != "real_observations_only"):
    raise RuntimeError("Expected a schema-1, real-observations-only, source-only release")
if chosen["kind"] == "root":
    SOURCE_ROOT = chosen["path"]
else:
    SOURCE_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="skyguard-source-"))
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
                raise RuntimeError("source archive must not contain symbolic links")
            destination = SOURCE_ROOT.joinpath(*parts.parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                if rel in extracted:
                    raise RuntimeError("duplicate source archive member: " + rel)
                extracted.add(rel)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
_verify_files(SOURCE_ROOT, SOURCE_MANIFEST.get("sha256"))
mismatches = [rel for rel, expected in CODE_MANIFEST["files"].items()
              if SOURCE_MANIFEST["sha256"].get(rel) != expected]
if mismatches:
    raise RuntimeError("Stale or missing source; regenerate source bundle and both notebooks: " + repr(mismatches))
for module_name in list(sys.modules):
    if module_name == "awsad" or module_name.startswith("awsad."):
        del sys.modules[module_name]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
importlib.invalidate_caches()
import awsad
assert pathlib.Path(awsad.__file__).resolve() == (SOURCE_ROOT / "src/awsad/__init__.py").resolve()
BUILD_UTC = datetime.now(timezone.utc)
BUILD_ID = BUILD_UTC.strftime("%Y%m%dT%H%M%S%fZ") + "-" + SOURCE_MANIFEST_SHA256[:12]
OUT = WORKDIR / ("awsad_real_bundle_" + BUILD_ID)
OUT.mkdir()
ARCHIVE = pathlib.Path(tempfile.mkdtemp(prefix="skyguard-surfrad-"))
print("Verified source release:", SOURCE_MANIFEST_SHA256)
print("Expected training source:", CODE_MANIFEST["sha256"])
print("Output directory:", OUT)
'''

ACQUIRE = r'''from awsad.data.fetch_surfrad import fetch_surfrad
STATIONS = ("bon", "fpk", "gwn")
YEARS = (2023, 2024)
inventory = fetch_surfrad(ARCHIVE, stations=STATIONS, years=YEARS, workers=4)
if not inventory.get("completed_at_utc") or inventory.get("failures") or not inventory.get("files"):
    raise RuntimeError("Acquisition is incomplete; inspect surfrad_acquisition.json before rebuilding")
print("Acquired published daily files:", len(inventory["files"]))
'''

BUILD = r'''from awsad.data.surfrad import build_surfrad_dataset
from awsad.data.training_contract import require_eligible_training_data
PROCESSED = OUT / "data" / "processed"
metadata = build_surfrad_dataset(ARCHIVE, PROCESSED, station_codes=STATIONS, years=YEARS)
preflight = require_eligible_training_data(PROCESSED, report_path=OUT / "data_preflight.json")
if not preflight.get("eligible"):
    raise RuntimeError("Independent raw-to-processed verification did not accept this dataset")
print("Original provider observation rows:", metadata["rows"])
print("Data eligibility:", preflight["eligible"])
print("Hardware fault labels remain unknown; provider quality flags are retained separately.")
'''

PACKAGE = r'''# Preserve the adapter's internal metadata and evidence hashes unchanged.
_verify_files(SOURCE_ROOT, SOURCE_MANIFEST["sha256"])
for rel in SOURCE_MANIFEST["sha256"]:
    destination = OUT.joinpath(*_relative_path(rel).parts)
    if destination.exists():
        raise RuntimeError("Source release collides with built data: " + rel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_ROOT / rel, destination)
provenance = OUT / "provenance"
provenance.mkdir(exist_ok=True)
if (provenance / "source_manifest.json").exists() or (provenance / "expected_source.json").exists():
    raise RuntimeError("Source release collides with reserved provenance paths")
(provenance / "source_manifest.json").write_bytes(SOURCE_MANIFEST_BYTES)
(provenance / "expected_source.json").write_text(json.dumps(CODE_MANIFEST, indent=2) + "\n", encoding="utf-8")
hashes = {path.relative_to(OUT).as_posix(): _sha256(path) for path in sorted(OUT.rglob("*")) if path.is_file()}
bundle_manifest = {
    "schema_version": 1, "project": "SkyGuard AI / SIH26073", "bundle_role": "processed_training_bundle",
    "training_data_policy": "real_observations_only", "build_id": BUILD_ID,
    "build_utc": BUILD_UTC.isoformat(), "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
    "training_source_sha256": CODE_MANIFEST["sha256"], "source": "noaa_surfrad",
    "stations": list(STATIONS), "years": list(YEARS), "observed_rows": metadata["rows"],
    "unknown_fault_labels": True, "processed_directory": "data/processed",
    "raw_observations_retained": True, "sha256": hashes,
}
(OUT / "MANIFEST.json").write_text(json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8")
_verify_files(OUT, hashes)
print("Verified output files:", len(hashes))
print("Publish this output directory as skyguard-sih26073-processed:", OUT)
print("Then run the matching offline CPU training notebook.")
'''


def build_notebook(root: Path = ROOT) -> dict:
    manifest = source_manifest(root)
    specs = [
        ("markdown", """# SkyGuard AI — original-observation dataset builder

This **online, CPU-only** builder acquires NOAA SURFRAD files published for
Bondville (`bon`), Fort Peck (`fpk`) and Goodwin Creek (`gwn`) in 2023–2024.
The provider documents temperature, independently measured relative humidity,
and station pressure. These US stations are a sensor-method development corpus;
they do not establish performance on Indian AWS.

The attached `skyguard-sih26073` source-only release is verified before import
or acquisition. The adapter retains native daily files, HTTP receipts, immutable
hashes, field provenance, quality codes and observed minute-00 records. Missing
readings stay missing and all hardware-fault labels stay unknown. The independent
preflight verifies the processed records against archived original observations.

The release chain remains: local source bundle → `skyguard-sih26073` → this
online builder → `skyguard-sih26073-processed` → offline CPU training. The output
is one complete directory, with source and original evidence. Publish the
directory after this notebook succeeds. Runtime depends on provider throughput;
an incomplete acquisition or failed data verification stops the build.
"""),
        ("markdown", "## 1 · Verify the attached source release"),
        ("code", "CODE_MANIFEST = " + repr(manifest) + "\n" + BOOTSTRAP),
        ("markdown", "## 2 · Acquire published original records\n\nEvery daily URL must first appear in the provider inventory. "
         "Four concurrent requests are the maximum. Network failures are recorded and cannot become invented rows."),
        ("code", ACQUIRE),
        ("markdown", "## 3 · Build and independently verify observations\n\nSelect existing minute-00 records; do not interpolate, "
         "aggregate, replace missing measurements or manufacture fault labels."),
        ("code", BUILD),
        ("markdown", "## 4 · Preserve the release chain and publishable output\n\nThe final manifest covers every retained file. "
         "Original provider evidence and the exact source release travel with the processed dataset."),
        ("code", PACKAGE),
    ]
    cells = []
    for index, (kind, source) in enumerate(specs):
        cell = {"cell_type": kind, "id": f"skyguard-builder-{index:02d}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            compile(source, f"builder_cell_{index}", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "skyguard": {"training_source_sha256": manifest["sha256"], "data_policy": "real_observations_only"},
            "kaggle": {"accelerator": "none", "dataSources": [], "dockerImageVersionId": None,
                       "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def build_local_bundle(source_bundle: str | Path, archive_dir: str | Path,
                       output_root: str | Path, *, stations=("bon", "fpk", "gwn"),
                       years=(2023, 2024)) -> Path:
    """Execute the notebook's verification/build/package cells without fetching.

    Accept an unpacked source release or its ZIP and an already acquired native
    SURFRAD archive. Run in a fresh process: like the notebook bootstrap, this
    selects the attached release as the active ``awsad`` import source.
    """
    import shutil
    import tempfile

    source = Path(source_bundle).resolve()
    archive = Path(archive_dir).resolve()
    output = Path(output_root).resolve()
    if not (archive / "surfrad_acquisition.json").is_file():
        raise FileNotFoundError("Expected an acquired SURFRAD archive with surfrad_acquisition.json")
    if not source.is_dir() and not (source.is_file() and source.suffix.lower() == ".zip"):
        raise ValueError("source_bundle must be an unpacked source release or ZIP")
    with tempfile.TemporaryDirectory(prefix="skyguard-local-input-") as temporary:
        inputs = source
        if source.is_file():
            inputs = Path(temporary)
            shutil.copy2(source, inputs / source.name)
        bootstrap = BOOTSTRAP.replace('pathlib.Path("/kaggle/working")', "pathlib.Path(" + repr(str(output)) + ")")
        bootstrap = bootstrap.replace('pathlib.Path("/kaggle/input")', "pathlib.Path(" + repr(str(inputs)) + ")")
        scope = {"CODE_MANIFEST": source_manifest(), "__name__": "skyguard_local_builder"}
        exec(compile(bootstrap, "local_source_bootstrap", "exec"), scope)
        scope["ARCHIVE"].rmdir()  # The just-created acquisition cache is still empty.
        scope.update(ARCHIVE=archive, STATIONS=tuple(stations), YEARS=tuple(years))
        exec(compile(BUILD, "local_observation_build", "exec"), scope)
        exec(compile(PACKAGE, "local_bundle_package", "exec"), scope)
        return scope["OUT"]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, help="Explicitly build locally from this source release")
    parser.add_argument("--archive-dir", type=Path, help="Completed native SURFRAD acquisition; never fetches")
    parser.add_argument("--output-root", type=Path, help="Parent directory for a new immutable processed bundle")
    args = parser.parse_args()
    if any((args.source_bundle, args.archive_dir, args.output_root)):
        if not all((args.source_bundle, args.archive_dir, args.output_root)):
            parser.error("local building requires --source-bundle, --archive-dir and --output-root")
        print("Local processed bundle:", build_local_bundle(args.source_bundle, args.archive_dir, args.output_root))
        return
    notebook = build_notebook()
    raw = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    for out in (ROOT / "kaggle/aws_data_builder.ipynb", ROOT / "kaggle_builder/aws_data_builder.ipynb"):
        out.write_text(raw, encoding="utf-8")
        print(f"Notebook written: {out} ({out.stat().st_size:,} bytes)")
    print("Training source SHA-256:", notebook["metadata"]["skyguard"]["training_source_sha256"])
    print("No acquisition or Kaggle job was launched.")


if __name__ == "__main__":
    main()
