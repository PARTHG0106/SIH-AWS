"""Create a deterministic, explicit-allowlist SIH submission ZIP (never publish).

Use --check to inspect readiness without writing an archive. No workspace walk,
environment file, credential, log, model search or implicit data discovery occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MINUTE = "artifacts_minute_20260928"
REPLAY = "artifacts_minute_fresh_20260928"
PATTERN = "artifacts_sih_20260930"
INDIA = "data/indian_demo_20260929"
ORIGINALS = "data/raw/surfrad_2025_january"
STATIONS = ("bon", "fpk", "gwn")
INDIAN_STATIONS = ("42182099999", "42348099999", "42410099999", "42809099999", "43057099999", "43279099999")

# The entire exact Python package is retained because the frozen protocol seals
# its file set. Legacy modules remain inert; no legacy dataset builder is shipped.
SOURCE_FILES = tuple("src/awsad/" + name for name in """
__init__.py correction.py explain.py health.py inference.py live_detector.py
minute_detection.py real_observation_pipeline.py station_health.py streaming.py train_pipeline.py
benchmark/__init__.py benchmark/consistency.py benchmark/fault_classifier.py
benchmark/injection_eval.py benchmark/operational_model.py benchmark/pattern_heads.py benchmark/scenarios.py benchmark/sih_evaluation.py
data/acquisition.py data/download_noaa_isd.py data/download_openmeteo.py data/fetch_surfrad.py
data/parse_isd.py data/parse_openmeteo.py data/stations.py data/surfrad_native.py data/surfrad.py
data/training_contract.py data/verify_surfrad.py demo/__init__.py demo/indian_stations.py demo/operational_feed.py
evaluation/calibration.py evaluation/metrics.py evaluation/nab.py evaluation/real_events.py evaluation/thresholds.py
models/ensemble.py models/fault_classifier.py models/isolation_forest.py models/lstm_autoencoder.py
models/lstm_forecaster.py models/spatial.py models/statistical.py models/transformer_ae.py
preprocessing/anomaly_injection.py preprocessing/features.py preprocessing/qc_rules.py utils/config.py
""".split())
FRONTEND_FILES = tuple("frontend/" + name for name in """
package.json package-lock.json index.html tsconfig.json vite.config.ts
src/App.tsx src/Chart.tsx src/api.ts src/main.tsx src/palette.ts src/theme.css src/ui.tsx
src/pages/BenchmarkPage.tsx src/pages/IndiaPage.tsx src/pages/LivePage.tsx src/pages/UsaPage.tsx
""".split())
APPLICATION_FILES = tuple("app/" + name for name in ("api.py", "live_api.py", "real_dashboard.py", "indian_dashboard.py", "streamlit_app.py", "theme.py"))
SCRIPT_FILES = tuple("scripts/" + name for name in """
package_sih_release.py serve_dashboard.py run_sih_benchmark.py run_minute_detection.py
seal_sih_model.py
select_sih_model.py
fetch_real_surfrad.py fetch_sih_holdout.py verify_minute_artifacts.py verify_live_detector.py
evaluate_real_events.py build_indian_demo.py
""".split())
TEST_FILES = tuple("tests/" + name for name in """
test_sih_package.py test_sih_scenarios.py test_sih_protocol.py test_operational_feed.py test_minute_detection.py
test_minute_runner.py test_live_detector.py test_live_api.py test_station_health.py
test_real_dashboard.py test_indian_dashboard.py test_indian_demo.py test_real_events.py
test_reviewed_event_cli.py test_acquisition.py test_surfrad.py test_surfrad_native.py
test_surfrad_portability.py test_edge_watchdog.py
test_pattern_heads.py test_sih_runtime_review.py
""".split())
DOCUMENT_FILES = tuple("docs/" + name for name in """
SIH_DEPLOYMENT.md SIH_REQUIREMENTS.md SIH_USE_CASES.md SIH_COMPLETION_PLAN.md SIH_FINAL_RESULTS_20261001.md
INDIAN_STATION_DEMO.md LIVE_DETECTION_20260930.md MINUTE_DETECTION_20260929.md
REAL_DATA_RESEARCH.md LICENSES.md
research/competitors_20260930/RESEARCH.md research/competitors_20260930/source_index.json
research/competitors_20260930/audit_receipts.py research/competitors_20260930/collect_sources.py
research/surfrad_20260926/VERIFICATION.md research/surfrad_events_20260928/VERIFICATION.md
research/surfrad_events_20260928/source_manifest.json research/surfrad_events_20260928/event_evidence_registry.json
""".split())
EDGE_FILES = ("edge/watchdog.py", "edge/run_watchdog.py", "edge/export_policy.py", "edge/README.md")
STATIC_FILES = ("README.md", "requirements.txt", "pytest.ini", *SOURCE_FILES, *FRONTEND_FILES,
                *APPLICATION_FILES, *SCRIPT_FILES, *TEST_FILES, *DOCUMENT_FILES, *EDGE_FILES)
MINUTE_FILES = ("models.joblib", "config.json", "detector.json", "metrics.json", "provenance.json",
                "artifact_verification.json", "independent_review_evaluation.json", "candidate_events.csv", "review_template.csv")
PATTERN_FILES = ("pattern_model.joblib", "isolation_baseline.joblib", "frozen.json", "metrics.json",
                 "selection.json", "calibration.json", "final_test.json")
OPTIONAL_FILES = (
    "artifacts_live_verified_20260930/metrics.json",
    "artifacts_edge_20260930/frozen_policy_v2.json",
    "artifacts_edge_20260930/host_measurement_default_v2.json",
    "artifacts_edge_20260930/host_measurement_frozen_v2.json",
)
LICENSE_FILES = {
    "frontend/node_modules/react/LICENSE": "third_party/licenses/react-LICENSE",
    "frontend/node_modules/react-dom/LICENSE": "third_party/licenses/react-dom-LICENSE",
    "frontend/node_modules/scheduler/LICENSE": "third_party/licenses/scheduler-LICENSE",
    "frontend/node_modules/echarts/LICENSE": "third_party/licenses/echarts-LICENSE",
    "frontend/node_modules/zrender/LICENSE": "third_party/licenses/zrender-LICENSE",
    "frontend/node_modules/tslib/LICENSE.txt": "third_party/licenses/tslib-LICENSE.txt",
}
RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "pyarrow",
                    "PyYAML", "tqdm", "matplotlib", "streamlit", "starlette", "uvicorn", "pytest")
HASH = re.compile(r"[0-9a-f]{64}\Z")
ASSET = re.compile(r"assets/[A-Za-z0-9_.-]+-[A-Za-z0-9_-]{6,}\.(?:js|css|woff2?|ttf|svg|png|jpe?g|webp|ico)\Z")
UNSET = object()
SOURCE_SEAL_FILES = (*SOURCE_FILES, "scripts/run_sih_benchmark.py", "scripts/run_minute_detection.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def safe_file(root, name, *, license_source=False):
    """Reject escapes, symlinks/junctions and credential-like paths before reading."""
    if not isinstance(name, str):
        raise ValueError("package paths must be strings")
    rel = PurePosixPath(name)
    if not isinstance(name, str) or "\\" in name or ":" in name or rel.is_absolute() or ".." in rel.parts:
        raise ValueError("unsafe package path: " + str(name))
    forbidden = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
    if any(part in forbidden for part in rel.parts) and not (license_source and name in LICENSE_FILES):
        raise ValueError("excluded package path: " + name)
    if any(part.startswith(".env") for part in rel.parts) or re.search(r"(?:kaggle.*\.json|credentials\.json|.*\.(?:pem|key|log))$", rel.name, re.I):
        raise ValueError("credential/environment/log path is never packageable: " + name)
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("symlink/junction is not packageable: " + name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("package path escaped workspace")
    return path


class _HTMLAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href") and value and not value.startswith(("https://", "http://", "data:", "#")):
                self.references.append(value)


@dataclass
class ReleasePlan:
    root: Path
    files: dict
    generated: dict
    errors: list
    warnings: list
    options: dict

    def report(self):
        return {"ready": not self.errors, "archive_created": False,
                "errors": self.errors, "warnings": self.warnings, "options": self.options,
                "files": len(self.files) + len(self.generated),
                "source_bytes": sum(item["bytes"] for item in self.files.values()),
                "replay_data_included": self.options["include_data"],
                "dashboard_ready_with_bundled_data": not self.errors and self.options["include_data"],
                "model_policy": "synthetic-pattern probabilities; real hardware status remains unknown"}


def make_plan(workspace=ROOT, *, include_data=False, include_india=False, include_originals=False,
              pattern_dir=PATTERN):
    root = Path(workspace).resolve()
    plan = ReleasePlan(root, {}, {}, [], [], {"include_data": include_data, "include_india": include_india,
                                           "include_originals": include_originals})
    try:
        # Only a caller-selected directory inside this workspace is accepted.
        # Its content still comes exclusively from PATTERN_FILES, never a walk.
        pattern_source = os.fspath(pattern_dir).replace("\\", "/")
        parts = PurePosixPath(pattern_source).parts
        if (not parts or PurePosixPath(pattern_source).is_absolute()
                or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in parts)):
            raise ValueError("--pattern-dir must be an explicit nonhidden workspace-relative artifact directory")
        pattern_source = PurePosixPath(pattern_source).as_posix()
        if not safe_file(root, pattern_source).is_dir():
            raise ValueError("selected pattern artifact directory is missing: " + pattern_source)
    except (TypeError, ValueError, OSError) as exc:
        plan.errors.append(str(exc))
        return plan
    plan.options["pattern_dir"] = pattern_source

    def add(name, *, expected=UNSET, required=True, target=None, category="source", license_source=False):
        try:
            path = safe_file(root, name, license_source=license_source)
            if not path.is_file():
                if required:
                    plan.errors.append("missing required file: " + name)
                return None
            digest = sha256(path)
            if expected is not UNSET and (not isinstance(expected, str) or not HASH.fullmatch(expected) or digest != expected):
                plan.errors.append("hash mismatch: " + name)
                return None
            destination = target or name
            safe_file(root, destination)
            plan.files[destination] = {"source": name, "sha256": digest, "bytes": path.stat().st_size, "category": category}
            return path
        except (ValueError, OSError) as exc:
            plan.errors.append(str(exc))
            return None

    def read(name):
        try:
            return json.loads(safe_file(root, name).read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            plan.errors.append(f"cannot read required JSON {name}: {exc}")
            return {}

    for name in STATIC_FILES:
        add(name)
    for name in OPTIONAL_FILES:
        add(name, required=False, category="verification_evidence")
    for source, target in LICENSE_FILES.items():
        add(source, target=target, category="third_party_license", license_source=True)

    # Frontend output is a referenced asset graph, never a directory wildcard.
    index = add("frontend/dist/index.html", category="frontend_build")
    if index:
        parser = _HTMLAssets()
        parser.feed(index.read_text(encoding="utf-8"))
        pending = [reference.lstrip("/") for reference in parser.references]
        seen = set()
        if not any(reference.endswith(".js") for reference in pending):
            plan.errors.append("frontend build has no JavaScript entry")
        while pending:
            reference = pending.pop()
            if reference in seen:
                continue
            seen.add(reference)
            if not ASSET.fullmatch(reference):
                plan.errors.append("frontend asset is outside the strict hashed-asset allowlist: " + reference)
                continue
            asset = add("frontend/dist/" + reference, category="frontend_build")
            if asset and asset.suffix in (".js", ".css"):
                for quoted in re.findall(r'''["'(]((?:\./|/assets/)[A-Za-z0-9_.\-/]+\.(?:js|css|woff2?|ttf|svg|png|jpe?g|webp|ico))["')]''', asset.read_text(encoding="utf-8")):
                    pending.append(quoted.lstrip("/") if quoted.startswith("/") else "assets/" + quoted[2:])
        source_times = [safe_file(root, name).stat().st_mtime_ns for name in FRONTEND_FILES if safe_file(root, name).is_file()]
        if source_times and index.stat().st_mtime_ns < max(source_times):
            plan.errors.append("frontend build is older than frontend source/config; rebuild it")

    for directory in (MINUTE, REPLAY):
        for filename in MINUTE_FILES:
            add(directory + "/" + filename, category="frozen_minute_model_or_provenance")
        provenance = read(directory + "/provenance.json")
        for relative, digest in provenance.get("source_files", {}).items():
            if relative not in (*SOURCE_FILES, "scripts/run_minute_detection.py"):
                plan.errors.append("unreviewed frozen snapshot path: " + relative)
                continue
            add(directory + "/source_snapshot/" + relative, expected=digest, category="frozen_source_snapshot")
        verification = read(directory + "/artifact_verification.json")
        if verification.get("status") != "verified":
            plan.errors.append("minute artifact verification is not complete: " + directory)
        for filename in ("models.joblib", "metrics.json", "provenance.json"):
            add(directory + "/" + filename, expected=verification.get(filename.split(".")[0] + "_sha256"),
                category="frozen_minute_model_or_provenance")
        detector = read(directory + "/detector.json")
        if detector.get("dataset_policy") != "real_observations_only":
            plan.errors.append("minute artifact policy is not real observations: " + directory)

    def add_pattern(filename, **kwargs):
        return add(pattern_source + "/" + filename, target=PATTERN + "/" + filename,
                   category="synthetic_pattern_model_or_metrics", **kwargs)

    for filename in PATTERN_FILES:
        add_pattern(filename)
    freeze, metrics, final = (read(pattern_source + "/" + name) for name in ("frozen.json", "metrics.json", "final_test.json"))
    if freeze.get("policy") != "synthetic_scenarios_on_original_observations":
        plan.errors.append("missing or unexpected frozen synthetic model policy")
    if metrics.get("status") != "final_test_complete" or final.get("status") != "final_test_complete":
        plan.errors.append("mandatory final scenario evaluation is incomplete")
    if metrics.get("final_test") != final:
        plan.errors.append("scenario metrics and final evaluation disagree")
    if final.get("frozen_sha256") != plan.files.get(PATTERN + "/frozen.json", {}).get("sha256"):
        plan.errors.append("final scenario evaluation does not match the frozen manifest")
    if final.get("status") == "final_test_complete":
        receipt_name = str(final.get("consumption_receipt", "")).replace("\\", "/")
        if not re.fullmatch(r"artifacts_sih_test_registry/[0-9a-f]{64}\.json", receipt_name):
            plan.errors.append("final evaluation requires its explicit holdout-consumption receipt")
        else:
            add(receipt_name, category="independent_test_consumption_receipt")
            receipt = read(receipt_name)
            if (receipt.get("status") != "completed" or receipt.get("frozen_sha256") != final.get("frozen_sha256")
                    or receipt.get("final_result_sha256") != plan.files.get(PATTERN + "/final_test.json", {}).get("sha256")):
                plan.errors.append("holdout-consumption receipt does not verify the final evaluation")
    add_pattern("pattern_model.joblib", expected=freeze.get("model_sha256"))
    add_pattern("isolation_baseline.joblib", expected=freeze.get("isolation_baseline_sha256"))
    fingerprints = freeze.get("source_fingerprints", {})
    if set(fingerprints) != set(SOURCE_SEAL_FILES):
        plan.errors.append("frozen protocol source set differs from the reviewed explicit allowlist")
    for name, expected in fingerprints.items():
        if name in SOURCE_SEAL_FILES:
            add(name, expected=expected)
    if "selection_script_sha256" in freeze:
        add("scripts/select_sih_model.py", expected=freeze["selection_script_sha256"])
    if "seal_script_sha256" in freeze:
        add("scripts/seal_sih_model.py", expected=freeze["seal_script_sha256"])
    for name, expected in freeze.get("frozen_minute_baseline", {}).items():
        if name not in ("models.joblib", "detector.json", "config.json"):
            plan.errors.append("unexpected frozen baseline artifact: " + name)
        else:
            add(MINUTE + "/" + name, expected=expected, category="frozen_minute_model_or_provenance")

    if include_data:
        provenance = read(REPLAY + "/provenance.json")
        expected_paths = {f"scored_observations/{station}_2025_01.parquet" for station in STATIONS}
        shards = provenance.get("scored_shards", [])
        if {item.get("file") for item in shards} != expected_paths:
            plan.errors.append("replay provenance must name exactly the three January2025 shards")
        for item in shards:
            if item.get("file") in expected_paths:
                add(REPLAY + "/" + item["file"], expected=item.get("sha256"), category="unchanged_january_replay")
    else:
        plan.warnings.append("Replay measurements omitted: launcher requires --replay-dir or a package built with --include-data.")

    if include_india:
        add(INDIA + "/manifest.json", category="separate_indian_demo")
        manifest = read(INDIA + "/manifest.json")
        if manifest.get("artifact_kind") != "indian_station_demo" or manifest.get("eligible_for_real_training") is not False:
            plan.errors.append("Indian bundle is not the separate demonstration artifact")
        catalog = manifest.get("catalog_file", {})
        if catalog.get("path") != "station_catalog.csv":
            plan.errors.append("unexpected Indian station catalog path")
        add(INDIA + "/station_catalog.csv", expected=catalog.get("sha256"), category="separate_indian_demo")
        expected_paths = {f"baseline_observations/{station}_2024.parquet" for station in INDIAN_STATIONS}
        if {item.get("path") for item in manifest.get("station_files", [])} != expected_paths:
            plan.errors.append("Indian baselines must be exactly the six reviewed station files")
        for item in manifest.get("station_files", []):
            if item.get("path") in expected_paths:
                add(INDIA + "/" + item["path"], expected=item.get("sha256"), category="separate_indian_demo")

    if include_originals:
        if not include_data:
            plan.errors.append("--include-originals requires --include-data")
        acquisition_path = ORIGINALS + "/surfrad_acquisition.json"
        evidence = read(REPLAY + "/provenance.json").get("native_manifest", {}).get("evidence", {})
        add(acquisition_path, expected=evidence.get("manifest_sha256"), category="original_acquisition")
        acquisition = read(acquisition_path)
        scope = acquisition.get("date_range", {})
        if acquisition.get("years") != [2025] or set(acquisition.get("stations", [])) != set(STATIONS) or scope != {
                "start": "2025-01-01T00:00:00+00:00", "end": "2025-02-01T00:00:00+00:00"}:
            plan.errors.append("original acquisition scope must be exactly January2025 / bon,fpk,gwn")
        entries = [*acquisition.get("documents", {}).values(), *acquisition.get("inventories", []), *acquisition.get("files", [])]
        for item in entries:
            filename, digest = item.get("file", ""), item.get("sha256", "")
            if not isinstance(digest, str) or not HASH.fullmatch(digest) or not re.fullmatch(re.escape(digest) + r"\.(?:dat|txt|html|raw)", filename):
                plan.errors.append("original acquisition path is not a hash-addressed permitted file")
                continue
            add(ORIGINALS + "/" + filename, expected=digest, category="original_acquisition")
        if include_india:
            manifest = read(INDIA + "/manifest.json")
            permitted = {f"data/raw/noaa_isd/2024/{station}.csv" for station in INDIAN_STATIONS}
            if {item.get("path") for item in manifest.get("raw_sources", [])} != permitted:
                plan.errors.append("Indian original-source paths differ from the six reviewed station files")
            for item in manifest.get("raw_sources", []):
                if item.get("path") in permitted:
                    add(item["path"], expected=item.get("sha256"), category="indian_original_acquisition")
            metadata = manifest.get("source_metadata", {})
            if metadata.get("path") != "data/raw/noaa_isd/isd-history.csv":
                plan.errors.append("unexpected Indian station metadata path")
            else:
                add(metadata["path"], expected=metadata.get("sha256"), category="indian_original_acquisition")

    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            plan.errors.append("packaging runtime lacks required installed package: " + name)
    for name, frozen_version in freeze.get("environment", {}).get("packages", {}).items():
        if packages.get(name) != frozen_version:
            plan.errors.append("current runtime differs from the model's frozen dependency: " + name)
    settings = {"replay_directory": REPLAY, "pattern_directory": PATTERN, "india_directory": INDIA,
                "python_version": sys.version.split()[0], "options": plan.options,
                "host_default": "127.0.0.1", "port_default": 8501,
                "semantics": "January historical replay and separately calibrated synthetic-pattern evidence; no connected station or hardware-fault accuracy claim"}
    plan.generated = {
        "release_settings.json": encoded(settings),
        "requirements-runtime.txt": ("# Versions from the packaging runtime; model-critical pins checked against frozen.json.\n" +
                                      "\n".join(f"{name}=={value}" for name, value in sorted(packages.items())) + "\n").encode(),
        "launch_sih.py": LAUNCHER.encode("utf-8"),
        "THIRD_PARTY_ATTRIBUTION.md": ATTRIBUTION.encode("utf-8"),
    }
    return plan


def write_package(plan, destination=None):
    """Write only a validated plan; fixed ZIP metadata makes identical input deterministic."""
    if plan.errors:
        raise ValueError("release is not ready:\n" + "\n".join(plan.errors))
    entries = dict(plan.files)
    for name, content in plan.generated.items():
        entries[name] = {"source": "generated", "sha256": hashlib.sha256(content).hexdigest(),
                         "bytes": len(content), "category": "release_runtime_or_attribution"}
    manifest = encoded({"schema_version": 1, "archive_policy": "explicit_allowlist_v1", "options": plan.options,
                        "files": entries, "warnings": plan.warnings,
                        "inventory_self_hash": "Inventory excluded from its own entries; ZIP SHA256 is recorded beside the archive."})
    inventory_hash = hashlib.sha256(manifest).hexdigest()
    out = Path(destination or plan.root / "out_sih_release").resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"skyguard-sih26073-{inventory_hash[:12]}.zip"
    receipt = archive.with_suffix(".zip.sha256")
    if archive.exists() or receipt.exists():
        raise FileExistsError("preserve the existing release; use a new output directory")
    created = False
    try:
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
            created = True
            for name in sorted((*entries, "release_manifest.json")):
                safe_file(plan.root, name)
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 30, 0, 0, 0))
                info.compress_type, info.create_system = zipfile.ZIP_DEFLATED, 3
                info.external_attr = 0o100644 << 16
                if name == "release_manifest.json":
                    handle.writestr(info, manifest)
                elif name in plan.generated:
                    handle.writestr(info, plan.generated[name])
                else:
                    entry = entries[name]
                    source = safe_file(plan.root, entry["source"], license_source=entry["source"] in LICENSE_FILES)
                    digest, size = hashlib.sha256(), 0
                    with source.open("rb") as incoming, handle.open(info, "w") as outgoing:
                        for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                            digest.update(chunk)
                            size += len(chunk)
                            outgoing.write(chunk)
                    if digest.hexdigest() != entry["sha256"] or size != entry["bytes"]:
                        raise ValueError("source changed after release check: " + entry["source"])
    except BaseException:
        if created and archive.is_file() and archive.parent == out:
            archive.unlink()  # Only this call's new incomplete archive, never an existing release.
        raise
    digest = sha256(archive)
    with receipt.open("x", encoding="ascii") as handle:
        handle.write(digest + "  " + archive.name + "\n")
    return {"archive": str(archive), "sha256": digest, "bytes": archive.stat().st_size,
            "inventory_sha256": inventory_hash, "files": len(entries) + 1,
            "replay_data_included": plan.options["include_data"], "published": False}


LAUNCHER = '''"""Verify the submission inventory and launch the local SkyGuard dashboard."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true")
parser.add_argument("--port", type=int, default=8501)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--replay-dir", type=Path)
parser.add_argument("--india-dir", type=Path)
args = parser.parse_args()
manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
for name, record in manifest["files"].items():
    path = root / name
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise SystemExit("Missing or unsafe release file: " + name)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != record["sha256"]:
        raise SystemExit("Release hash mismatch: " + name)
settings = json.loads((root / "release_settings.json").read_text(encoding="utf-8"))
freeze = json.loads((root / settings["pattern_directory"] / "frozen.json").read_text(encoding="utf-8"))
if sys.version.split()[0] != settings["python_version"]:
    raise SystemExit("Use the packaged Python version: " + settings["python_version"])
for name, expected in freeze.get("environment", {}).get("packages", {}).items():
    try:
        actual = version(name)
    except PackageNotFoundError:
        raise SystemExit("Missing dependency; install requirements-runtime.txt: " + name)
    if actual != expected:
        raise SystemExit("Frozen model dependency mismatch; install requirements-runtime.txt: " + name)
replay = args.replay_dir or root / settings["replay_directory"]
if not all((replay / "scored_observations" / (station + "_2025_01.parquet")).is_file() for station in ("bon", "fpk", "gwn")):
    raise SystemExit("Replay data are absent. Build with --include-data or pass --replay-dir with the verified January bundle.")
os.environ["SKYGUARD_ARTIFACTS"] = str(replay.resolve())
os.environ["SKYGUARD_SIH_BENCHMARK"] = str(root / settings["pattern_directory"])
os.environ["SKYGUARD_INDIAN_DEMO"] = str(args.india_dir or root / settings["india_directory"])
os.chdir(root)
sys.path[:0] = [str(root), str(root / "src")]
if args.check:
    print(json.dumps({"inventory_verified": True, "replay_directory": str(replay), "settings": settings}, indent=2))
else:
    import uvicorn
    uvicorn.run("app.api:app", host=args.host, port=args.port)
'''

ATTRIBUTION = '''# Submission attribution and scope

This archive combines material with different rights. It is not licensed as a single CC0 dataset.

- SkyGuard source/model artifacts: project authors retain rights; no project-wide open-source license has been selected.
- NOAA GML SURFRAD observations: CC0 1.0 as stated in the archived provider README. Acknowledge NOAA GML, Augustine, DeLuisi and Long (2000), BAMS 81,2341–2357; Augustine et al. (2005), J. Atmos. Oceanic Technol.22,1460–1472. No NOAA endorsement is implied. Original source hashes and provenance remain attached; published observations may be revised by the provider.
- Optional Indian baselines: NOAA NCEI Integrated Surface Database (Global Hourly). Their manifest preserves original attribution and the limits of the local-archive rights verification. Derived RH and sea-level pressure retain their meanings; demonstration scenarios are synthetic.
- React, React DOM, Scheduler and tslib: MIT; Apache ECharts: Apache-2.0; ZRender: BSD-3-Clause. Exact bundled frontend license texts are in third_party/licenses. The lockfile identifies versions.
- Python libraries are installed separately, not vendored. requirements-runtime.txt pins selected runtime packages and the synthetic model's frozen critical versions; it is not a complete environment image. Each library retains its own license.
- Inspected public research is represented by the comparison report and source_index.json citation/hash metadata, including original URLs/authors and any stated licenses. Full retrieved peer source text is not included. These are evidence references, not relicensed project code or imported implementations.
- Optional web fonts are requested from Google Fonts by the UI; no font files are redistributed here. The interface has local fallback fonts.

Real replay results, synthetic scenario metrics and edge policy checks remain separate. Unknown real hardware status is never converted into a known negative label. See docs/LICENSES.md and docs/SIH_DEPLOYMENT.md.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--pattern-dir", default=PATTERN,
                        help="explicit workspace-relative selected artifact directory; archived at artifacts_sih_20260930")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true", help="check readiness only; write no archive")
    parser.add_argument("--include-data", action="store_true", help="include exactly the three verified January replay shards")
    parser.add_argument("--include-india", action="store_true", help="include the six separate Indian demonstration baselines")
    parser.add_argument("--include-originals", action="store_true", help="include manifest-listed January originals, and Indian originals if selected")
    args = parser.parse_args()
    plan = make_plan(args.workspace, include_data=args.include_data, include_india=args.include_india,
                     include_originals=args.include_originals, pattern_dir=args.pattern_dir)
    if args.check or plan.errors:
        print(json.dumps(plan.report(), indent=2))
        return 0 if not plan.errors else 2
    print(json.dumps(write_package(plan, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
