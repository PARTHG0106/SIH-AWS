"""Generate an offline native-minute notebook with its exact source snapshot.

The observation release remains source -> builder -> processed bundle. Its
immutable originals are re-parsed by the separately fingerprinted minute code
embedded here. The legacy hourly notebook and its source checks are unchanged.
This generator prepares local files only; it never uploads or starts a run.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = "scripts/run_minute_detection.py"


def embedded_source(root: Path = ROOT) -> tuple[dict, str]:
    """Follow all package imports, including function-local and relative ones."""
    pending = [(ENTRYPOINT, None), ("src/awsad/__init__.py", "awsad")]
    seen, contents = set(), {}
    while pending:
        rel, module = pending.pop()
        if rel in seen:
            continue
        seen.add(rel)
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError("missing minute source: " + rel)
        raw = path.read_bytes()
        contents[rel] = base64.b64encode(raw).decode("ascii")
        package = [] if module is None else module.split(".")
        if path.name != "__init__.py" and package:
            package = package[:-1]
        modules = {".".join(package[:i]) for i in range(1, len(package) + 1)}
        for node in ast.walk(ast.parse(raw, filename=rel)):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                target = ".".join(prefix + (node.module.split(".") if node.module else []))
                modules.add(target)
                modules.update(target + "." + alias.name for alias in node.names if alias.name != "*")
        for name in modules:
            if name != "awsad" and not name.startswith("awsad."):
                continue
            stem = root / "src" / Path(*name.split("."))
            child = stem.with_suffix(".py")
            if not child.is_file():
                child = stem / "__init__.py"
            if child.is_file():
                pending.append((child.relative_to(root).as_posix(), name))
    required = {ENTRYPOINT, "src/awsad/__init__.py", "src/awsad/minute_detection.py",
                "src/awsad/data/surfrad_native.py", "src/awsad/data/surfrad.py",
                "src/awsad/data/acquisition.py", "src/awsad/data/fetch_surfrad.py",
                "src/awsad/data/verify_surfrad.py", "src/awsad/evaluation/real_events.py"}
    if not required <= set(contents):
        raise RuntimeError("incomplete minute source closure: " + repr(required - set(contents)))
    contents = dict(sorted(contents.items()))
    hashes = {rel: hashlib.sha256(base64.b64decode(value)).hexdigest()
              for rel, value in contents.items()}
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {"schema_version": 1, "role": "embedded_native_minute_source",
                "sha256": hashlib.sha256(canonical).hexdigest(), "files": hashes}
    raw_payload = json.dumps(contents, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = base64.b64encode(zlib.compress(raw_payload, 9)).decode("ascii")
    return manifest, payload


# These helpers are tested independently with metadata-only fixtures. They do
# not import awsad or any learning library before input/source integrity checks.
BOOTSTRAP_HELPERS = r'''import base64, hashlib, json, os, pathlib, shutil, sys, zipfile, zlib

def _relative_path(rel):
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel:
        raise RuntimeError("unsafe bundle path: " + str(rel))
    path = pathlib.PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != rel or rel == ".":
        raise RuntimeError("unsafe bundle path: " + rel)
    return path

def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _decode_manifest(raw):
    if len(raw) > 2_000_000:
        raise RuntimeError("oversized processed manifest")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise RuntimeError("processed manifest must be an object")
    return manifest

def _read_manifest(root):
    path = root / "MANIFEST.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
        raise RuntimeError("missing or invalid processed manifest")
    return _decode_manifest(path.read_bytes())

def locate_bundles(inputdir):
    candidates = []
    for metadata in sorted(inputdir.glob("**/data/processed/source_metadata.json")):
        root = metadata.parents[2]
        candidates.append({"kind": "root", "path": root, "prefix": "", "manifest": _read_manifest(root)})
    for archive in sorted(inputdir.glob("**/*.zip")):
        try:
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
                for name in names:
                    suffix = "data/processed/source_metadata.json"
                    if not name.endswith(suffix):
                        continue
                    prefix = name[:-len(suffix)]
                    if prefix:
                        _relative_path(prefix.rstrip("/"))
                    manifest_name = prefix + "MANIFEST.json"
                    if manifest_name not in names or zf.getinfo(manifest_name).file_size > 2_000_000:
                        raise RuntimeError("missing or oversized processed manifest")
                    candidates.append({"kind": "zip", "path": archive, "prefix": prefix,
                                       "manifest": _decode_manifest(zf.read(manifest_name))})
        except zipfile.BadZipFile:
            continue
    if not candidates:
        raise FileNotFoundError("Attach the SURFRAD processed bundle with immutable raw files. "
                                "Use the source -> builder -> processed dataset release flow.")
    identities = {hashlib.sha256(json.dumps(c["manifest"], sort_keys=True).encode()).hexdigest()
                  for c in candidates}
    if len(identities) != 1:
        raise RuntimeError("ambiguous processed inputs: attach exactly one immutable release")
    return sorted(candidates, key=lambda c: (c["kind"] != "root", str(c["path"])))[0]

def materialize_bundle(chosen, destination):
    if chosen["kind"] == "root":
        return chosen["path"]
    destination.mkdir(parents=True, exist_ok=False)
    extracted, total = set(), 0
    with zipfile.ZipFile(chosen["path"]) as zf:
        for member in zf.infolist():
            if member.orig_filename != member.filename:
                raise RuntimeError("unsafe bundle path spelling: " + member.orig_filename)
            if not member.filename.startswith(chosen["prefix"]):
                continue
            rel = member.filename[len(chosen["prefix"]):].rstrip("/")
            if not rel:
                continue
            parts = _relative_path(rel)
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError("bundle archives cannot contain symbolic links")
            path = destination.joinpath(*parts.parts)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            if rel in extracted:
                raise RuntimeError("duplicate bundle archive member: " + rel)
            extracted.add(rel)
            total += member.file_size
            if total > 10_000_000_000:
                raise RuntimeError("bundle exceeds 10 GB extraction limit")
            path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    return destination

def verify_processed_bundle(root, manifest):
    if (manifest.get("schema_version") != 1
            or manifest.get("bundle_role") != "processed_training_bundle"
            or manifest.get("training_data_policy") != "real_observations_only"
            or manifest.get("source") != "noaa_surfrad"
            or manifest.get("raw_observations_retained") is not True
            or manifest.get("unknown_fault_labels") is not True):
        raise RuntimeError("expected a real-observation SURFRAD processed release with original raw files")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("processed release requires a complete hash manifest")
    resolved = root.resolve()
    for rel, expected in hashes.items():
        parts = _relative_path(rel)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise RuntimeError("invalid expected SHA-256: " + rel)
        path = root.joinpath(*parts.parts)
        if (path.is_symlink() or not path.is_file() or resolved not in path.resolve().parents
                or _sha256_file(path) != expected):
            raise RuntimeError("processed file integrity failure: " + rel)
    paths = list(root.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise RuntimeError("processed release cannot contain symbolic links")
    actual = {p.relative_to(root).as_posix() for p in paths if p.is_file() and p != root / "MANIFEST.json"}
    if actual != set(hashes):
        raise RuntimeError("processed manifest does not cover exactly the retained files")
    source_manifest = root / "provenance/source_manifest.json"
    if (not source_manifest.is_file()
            or _sha256_file(source_manifest) != manifest.get("source_manifest_sha256")):
        raise RuntimeError("missing or changed original source-release provenance")
    return {"eligible": True, "build_id": manifest.get("build_id"), "checked_files": len(hashes),
            "processed_manifest_sha256": _sha256_file(root / "MANIFEST.json"),
            "original_source_manifest_sha256": manifest["source_manifest_sha256"],
            "original_hourly_training_source_sha256": manifest.get("training_source_sha256"),
            "role": "observation-release integrity; native measurement admission follows during reconstruction"}

def materialize_source(manifest, payload, destination):
    if manifest.get("schema_version") != 1 or manifest.get("role") != "embedded_native_minute_source":
        raise RuntimeError("unsupported embedded minute source schema")
    canonical = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest["sha256"]:
        raise RuntimeError("embedded source manifest integrity failure")
    contents = json.loads(zlib.decompress(base64.b64decode(payload, validate=True)))
    if set(contents) != set(manifest["files"]):
        raise RuntimeError("embedded source file set mismatch")
    decoded = {}
    for rel, encoded in contents.items():
        parts = _relative_path(rel)
        if not ((rel.startswith("src/awsad/") and rel.endswith(".py"))
                or rel == "scripts/run_minute_detection.py"):
            raise RuntimeError("unexpected embedded source path: " + rel)
        raw = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(raw).hexdigest() != manifest["files"][rel]:
            raise RuntimeError("embedded source hash mismatch: " + rel)
        decoded[rel] = (parts, raw)
    destination.mkdir(parents=True, exist_ok=False)
    for rel, (parts, raw) in decoded.items():
        path = destination.joinpath(*parts.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (destination / "CODE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination
'''


BOOTSTRAP = r'''from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import platform

sys.dont_write_bytecode = True
WORKDIR = pathlib.Path("/kaggle/working")
INPUTDIR = pathlib.Path("/kaggle/input")
WORKDIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + CODE_MANIFEST["sha256"][:10]
RUN_ROOT = WORKDIR / ("minute_" + RUN_ID)
RUN_ROOT.mkdir()
PREFLIGHT = None

# A unique directory on every run preserves previous experiments.
chosen = locate_bundles(INPUTDIR)
BUNDLE = materialize_bundle(chosen, RUN_ROOT / "observation_release")
release_report = verify_processed_bundle(BUNDLE, chosen["manifest"])
SOURCE_ROOT = materialize_source(CODE_MANIFEST, SOURCE_PAYLOAD, RUN_ROOT / "minute_source")
PROCESSED = BUNDLE / "data/processed"
CACHE = RUN_ROOT / "native_cache"
OUT = RUN_ROOT / "artifacts"

versions = {"python": platform.python_version()}
for distribution in ("numpy", "pandas", "scikit-learn", "pyarrow", "joblib", "threadpoolctl"):
    try:
        versions[distribution] = version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError("Missing dependency in this offline runtime: " + distribution) from exc
report = {"observation_release": release_report,
          "minute_source_sha256": CODE_MANIFEST["sha256"], "runtime_versions": versions,
          "source_policy": "Execute only the embedded minute source; archived hourly source is observation-release provenance",
          "observations_policy": "Only immutable original provider records; no manufactured readings or fault labels"}
(RUN_ROOT / "source_and_release_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
(RUN_ROOT / "processed_MANIFEST.json").write_bytes((BUNDLE / "MANIFEST.json").read_bytes())
PREFLIGHT = report
print("Verified observation build:", release_report["build_id"])
print("Embedded minute source:", CODE_MANIFEST["sha256"])
print("Native measurements will be verified against acquired originals as they are reconstructed.")
'''


RUN = r'''import subprocess, time
if PREFLIGHT is None:
    raise RuntimeError("Source and observation-release preflight must succeed first")
CONFIG_PATH = RUN_ROOT / "minute_config.json"
CONFIG_PATH.write_text(json.dumps(MINUTE_CONFIG, indent=2) + "\n")
command = [sys.executable, "-B", str(SOURCE_ROOT / "scripts/run_minute_detection.py"),
           "--archive-dir", str(PROCESSED), "--cache-dir", str(CACHE),
           "--out", str(OUT), "--config", str(CONFIG_PATH)]
environment = dict(os.environ)
environment["PYTHONDONTWRITEBYTECODE"] = "1"
environment["PYTHONNOUSERSITE"] = "1"
# The subprocess imports the isolated snapshot, never the older attached code.
environment["PYTHONPATH"] = str(SOURCE_ROOT / "src")
started = time.monotonic()
with (RUN_ROOT / "run.log").open("w", encoding="utf-8") as log:
    process = subprocess.Popen(command, cwd=SOURCE_ROOT, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end="")
    code = process.wait()
if code:
    raise RuntimeError("Minute pipeline failed; preserved diagnostics: " + str(RUN_ROOT / "run.log"))
elapsed = time.monotonic() - started
METRICS = json.loads((OUT / "metrics.json").read_text())
if METRICS.get("status") != "real_minute_replay_complete" or METRICS.get("synthetic_observations") != 0:
    raise RuntimeError("Unexpected minute pipeline completion status")
print("Run seconds:", round(elapsed, 1))
print(json.dumps({key: METRICS[key] for key in ("native_observations", "candidate_minutes",
      "candidate_event_proposals", "known_hardware_fault_labels")}, indent=2))
print("Candidate events remain unconfirmed. Fault precision/recall/F1 have not been established.")
'''


EXPORT = r'''# Archive enough source/configuration to reproduce this frozen development run.
# Original raw bytes remain in the attached processed release; hashes and
# release identity are included here. Reconstruct its native cache on rerun.
if not (OUT / "metrics.json").is_file() or not (OUT / "models.joblib").is_file():
    raise RuntimeError("A completed minute training run is required before export")
shutil.copytree(SOURCE_ROOT, OUT / "minute_source")
for name in ("minute_config.json", "source_and_release_preflight.json", "processed_MANIFEST.json", "run.log"):
    shutil.copy2(RUN_ROOT / name, OUT / name)
shutil.copy2(CACHE / "native_manifest.json", OUT / "native_manifest.json")
hashes = {p.relative_to(OUT).as_posix(): _sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
(OUT / "ARTIFACT_MANIFEST.json").write_text(json.dumps({
    "schema_version": 1, "role": "native_minute_detector_artifacts",
    "observation_build_id": PREFLIGHT["observation_release"]["build_id"],
    "minute_source_sha256": CODE_MANIFEST["sha256"], "sha256": hashes,
    "data_policy": "real_observations_only", "hardware_fault_labels": "unknown unless independently reviewed",
    "evaluation_scope": "development and excluded-station replay before 2024-11; not fresh-period evaluation"
}, indent=2) + "\n")
archive_path = WORKDIR / ("skyguard_minute_artifacts_" + RUN_ID + ".zip")
with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            zf.write(path, "artifacts/" + path.relative_to(OUT).as_posix())
print("Download artifacts:", archive_path)
print("Review candidate_events.csv and review_template.csv; do not treat unreviewed rows as fault-free.")
from IPython.display import FileLink, display
display(FileLink(str(archive_path)))
'''


def _cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
            "id": hashlib.sha256((kind + source).encode("utf-8")).hexdigest()[:16]}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebook(root: Path = ROOT) -> dict:
    manifest, payload = embedded_source(root)
    # Read defaults from source without importing its learning dependencies.
    tree = ast.parse((root / "src/awsad/minute_detection.py").read_text(encoding="utf-8"))
    config_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MinuteConfig")
    config = {n.target.id: ast.literal_eval(n.value) for n in config_class.body
              if isinstance(n, ast.AnnAssign) and n.value is not None}
    config = json.loads(json.dumps(config))
    constants = ("CODE_MANIFEST = " + repr(manifest) + "\n"
                 + "SOURCE_PAYLOAD = " + repr(payload) + "\n"
                 + "MINUTE_CONFIG = " + repr(config) + "\n")
    cells = [
        _cell("markdown", "# SkyGuard AI — native-minute anomaly candidates\n\n"
              "This offline CPU notebook reconstructs unchanged SURFRAD one-minute observations, "
              "fits past-only forecasts, calibrates five signals per channel, and exports traceable "
              "candidate events for independent review. Inputs are measured temperature (°C), station "
              "pressure (hPa), and independently measured RH (%). Missing records and readings stay missing.\n\n"
              "**Observation workflow:** source v19 → builder v18 → processed dataset v9, containing "
              "the original daily files. Attach `krishnagupta02468/skyguard-sih26073-processed`; no new "
              "source dataset is required to reuse those immutable observations. This notebook embeds "
              "a separately hashed snapshot of the new minute implementation. It does not execute the "
              "older hourly implementation archived in that release. A different processed release "
              "must pass the same integrity, provider-document, scope and raw-row admission checks.\n\n"
              "Use Kaggle CPU, Internet **off**, and run all cells. Reconstruction verifies every "
              "consumed daily original and may take several minutes. It processes station-month "
              "shards and bounded training samples; it does not load the full native corpus at once.\n"),
        _cell("code", constants),
        _cell("markdown", "## Verify the observation release and exact minute source\n\n"
              "The embedded code manifest is distinct from the older observation-release source manifest. "
              "Both are retained in the output. Hashes check integrity; the native parser also checks "
              "provider documentation, units, timestamps, raw quality codes, and raw-file provenance.\n"),
        _cell("code", BOOTSTRAP_HELPERS + "\n" + BOOTSTRAP),
        _cell("markdown", "## Fit, select, calibrate, then replay\n\n"
              "Training ends before July 2024. July–August select each forecast against persistence; "
              "September–October calibrate the frozen thresholds. Goodwin Creek (`gwn`) is excluded "
              "from fitting, model selection and calibration, and uses thresholds pooled from the "
              "seen stations. The previously examined November–December 2024 period is excluded.\n\n"
              "Signals are one-minute forecast residual, abrupt change, duration of exactly repeated "
              "readings, one-hour residual, and sustained deviation. Thresholds use provider-accepted "
              "reference observations; acceptance is not a known-normal label. The score is a signal "
              "exceedance measure, never a fault probability. Configuration is stored with the artifacts.\n"),
        _cell("code", RUN),
        _cell("markdown", "## Download reproducible artifacts\n\n"
              "The ZIP includes models, thresholds, native/scored manifests, original observation "
              "lineage, candidate events, an unreviewed event template, metrics, source, and run log. "
              "The native cache can be reconstructed from the identified attached observation release.\n\n"
              "**Evaluation limit:** these are unconfirmed anomaly candidates on real measurements. "
              "Provider QC is quality evidence, not hardware-fault truth. No confirmed event benchmark, "
              "fault precision/recall/F1, false-alarm rate, hardware diagnosis, repair, live connection, "
              "or Indian IMD AWS validation is claimed. Fresh-period evaluation must use the frozen "
              "detector and separately acquired observations without retuning.\n"),
        _cell("code", EXPORT),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "skyguard": {"purpose": "real_native_minute_detection", "source_sha256": manifest["sha256"]}},
        "nbformat": 4, "nbformat_minor": 5}


def main():
    raw = json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"
    for path in (ROOT / "kaggle/aws_minute_detection.ipynb", ROOT / "kaggle_minute/aws_minute_detection.ipynb"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw, encoding="utf-8", newline="\n")
        print(path)
    metadata = {"id": "krishnagupta02468/skyguard-ai-minute-detection-sih26073",
                "title": "SkyGuard AI - Minute Detection (SIH26073)",
                "code_file": "aws_minute_detection.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
                "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["krishnagupta02468/skyguard-sih26073-processed"],
                "competition_sources": [], "kernel_sources": [], "model_sources": [], "category_ids": []}
    (ROOT / "kaggle_minute/kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
