"""Notebook contract tests; fixtures are metadata only, never training data."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("training_notebook_generator", ROOT / "kaggle/build_notebook.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)
BUILDER_SPEC = importlib.util.spec_from_file_location("data_builder_generator", ROOT / "kaggle/build_data_builder.py")
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


def test_expected_source_fingerprints_match_training_dependencies_and_cells_compile():
    notebook = GENERATOR.build_notebook()
    code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    manifest = ast.literal_eval(ast.parse(code_cells[0]).body[0].value)
    payload = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
    sources = manifest["files"]
    assert "src/awsad/real_observation_pipeline.py" in sources
    assert "src/awsad/data/training_contract.py" in sources
    assert "src/awsad/data/surfrad.py" in sources
    assert "src/awsad/data/fetch_surfrad.py" in sources
    assert "src/awsad/data/acquisition.py" in sources
    assert "src/awsad/train_pipeline.py" not in sources
    assert "src/awsad/models/lstm_autoencoder.py" not in sources
    assert "src/awsad/data/download_noaa_isd.py" not in sources
    assert "src/awsad/preprocessing/anomaly_injection.py" not in sources
    assert "SOURCE_PAYLOAD" not in code_cells[0]
    for rel in sources:
        raw = (ROOT / rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest["files"][rel]
        assert rel.startswith("src/awsad/") and rel.endswith(".py")
    for source in code_cells:
        compile(source, "generated_notebook_cell", "exec")


def _run_bootstrap(tmp_path, *, stale, outer_hash_failure=False):
    """Exercise input/source selection in a separate interpreter."""
    bundle = tmp_path / "inputs" / "old_bundle"
    processed = bundle / "data" / "processed"
    processed.mkdir(parents=True)
    # Deliberately incomplete/legacy metadata, not invented observations.
    (processed / "source_metadata.json").write_text(json.dumps({
        "status": "legacy_local_snapshot",
        "rh_policy": "NOAA ISD RH is derived from TMP+DEW",
        "pressure_policy": "NOAA ISD SLP is sea-level pressure",
    }))
    for rel in GENERATOR.source_manifest()["files"]:
        destination = bundle / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    if stale:
        (bundle / "src/awsad/__init__.py").write_text("raise AssertionError('stale attached source imported')\n")
    if outer_hash_failure:
        provenance = bundle / "provenance/source_manifest.json"
        provenance.parent.mkdir()
        provenance.write_text("{}")
        manifest = {"schema_version": 1, "bundle_role": "processed_training_bundle",
                    "training_data_policy": "real_observations_only", "build_id": "test-outer-hash-failure",
                    "training_source_sha256": GENERATOR.source_manifest()["sha256"],
                    "source_manifest_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
                    "sha256": {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in bundle.rglob("*") if p.is_file()}}
        manifest["sha256"]["data/processed/source_metadata.json"] = "0" * 64
        (bundle / "MANIFEST.json").write_text(json.dumps(manifest))
    bootstrap = "".join(GENERATOR.build_notebook()["cells"][2]["source"])
    bootstrap = bootstrap.replace('pathlib.Path("/kaggle/working")', repr(tmp_path / "working"))
    bootstrap = bootstrap.replace('pathlib.Path("/kaggle/input")', repr(tmp_path / "inputs"))
    # pathlib's repr names platform-specific classes; construct them explicitly.
    script = "from pathlib import WindowsPath, PosixPath\nimport builtins, sys\n"
    script += "force_data_eligibility = " + repr(outer_hash_failure) + "\n"
    script += """
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'pandas', 'numpy'}:
        raise AssertionError('heavy/data import before rejection: ' + name)
    imported = original_import(name, *args, **kwargs)
    if force_data_eligibility and name == 'awsad.data.training_contract':
        # Isolate outer-release failure without inventing a training dataset.
        imported.require_eligible_training_data = lambda *a, **k: {'eligible': True}
    return imported
builtins.__import__ = guarded_import
try:
"""
    script += "\n".join("    " + line for line in bootstrap.splitlines())
    script += "\nexcept Exception as exc:\n"
    if stale:
        script += "    assert type(exc).__name__ == 'RuntimeError' and 'stale or missing training source' in str(exc), repr(exc)\n"
        script += "    assert 'awsad' not in sys.modules\n"
    elif outer_hash_failure:
        script += "    assert type(exc).__name__ == 'RuntimeError' and 'input bundle hash mismatch' in str(exc), repr(exc)\n"
        script += "    assert DATA_PREFLIGHT['eligible'] is True and PREFLIGHT is None\n"
    else:
        script += "    assert type(exc).__name__ == 'TrainingDataContractError', repr(exc)\n"
        script += "    assert exc.report['eligible'] is False\n"
    script += "    assert 'torch' not in sys.modules and 'pandas' not in sys.modules\n"
    script += "else:\n    raise AssertionError('ineligible input unexpectedly passed preflight')\n"
    if outer_hash_failure:
        script += "try:\n    exec(" + repr(GENERATOR.CONFIG) + ", globals())\n"
        script += "except RuntimeError as exc:\n    assert 'Data/source preflight must succeed' in str(exc), repr(exc)\n"
        script += "else:\n    raise AssertionError('configuration accepted partial preflight success')\n"
    check = tmp_path / "bootstrap_check.py"
    check.write_text(script, encoding="utf-8")
    result = subprocess.run([sys.executable, str(check)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path / "working"


def test_bootstrap_rejects_stale_processed_source_before_importing_it(tmp_path):
    working = _run_bootstrap(tmp_path, stale=True)
    source_reports = list(working.glob("artifacts_*/source_preflight.json"))
    assert len(source_reports) == 1
    report = json.loads(source_reports[0].read_text())
    assert report["eligible"] is False
    assert report["mismatches"][0]["path"] == "src/awsad/__init__.py"
    assert not list(working.glob("artifacts_*/data_preflight.json"))


def test_bootstrap_uses_processed_source_and_rejects_legacy_before_data_or_gpu(tmp_path):
    """Matching source permits inspection, never an override of data policy."""
    _run_bootstrap(tmp_path, stale=False)
    reports = list((tmp_path / "working").glob("artifacts_*/data_preflight.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["eligible"] is False
    assert not list((tmp_path / "working").glob("artifacts_*/metrics.json"))


def test_failed_outer_hash_does_not_publish_partial_eligibility_to_later_cells(tmp_path):
    _run_bootstrap(tmp_path, stale=False, outer_hash_failure=True)


def test_notebook_profiles_are_cpu_budgets_and_holdback_dates_are_fixed():
    """Inspect declared budgets and protocol without executing training."""
    config_ast = ast.parse(GENERATOR.CONFIG)
    selected = [node for node in config_ast.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "PROFILES" for target in node.targets)]
    env = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "profile_config", "exec"), env)
    profiles = env["PROFILES"]
    assert set(profiles) == {"fast", "standard", "full"}
    for parameter in ("max_iter", "max_train_rows", "max_validation_rows"):
        assert 0 < profiles["fast"][parameter] < profiles["standard"][parameter] < profiles["full"][parameter]
    config = next(node.value for node in config_ast.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets))
    values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in config.keywords if keyword.arg}
    assert values["train_end"] == "2024-09-01T00:00:00Z"
    assert values["validation_end"] == "2024-11-01T00:00:00Z"
    assert values["test_end"] == "2025-01-01T00:00:00Z"
    assert values["horizon_minutes"] == (60,)
    assert values["thinning_minutes"] == 60
    assert values["holdout_station_ids"] == ("gwn",)
    assert values["max_cpu_threads"] == 4
    assert values["random_seed"] == 42
    assert values["evaluate_test"] is True
    assert "torch" not in GENERATOR.CONFIG


def test_builder_and_training_notebooks_share_source_and_use_expected_environments():
    training = GENERATOR.build_notebook()
    builder = BUILDER.build_notebook()
    assert builder["metadata"]["skyguard"]["training_source_sha256"] == training["metadata"]["skyguard"]["training_source_sha256"]
    for notebook, online in ((training, False), (builder, True)):
        assert notebook["metadata"]["kaggle"]["accelerator"] == "none"
        assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is online
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "generated_cell", "exec")
    builder_code = "\n".join("".join(c["source"]) for c in builder["cells"] if c["cell_type"] == "code")
    assert "fetch_surfrad(ARCHIVE" in builder_code
    assert "build_surfrad_dataset(ARCHIVE" in builder_code
    assert "require_eligible_training_data(PROCESSED" in builder_code
    for forbidden in ("anomaly_injection", "download_noaa_isd", "openmeteo", "inject_", "impute_hourly"):
        assert forbidden not in builder_code
    for filename, online in (("kaggle/kernel-metadata.json", False), ("kaggle_builder/kernel-metadata.json", True)):
        metadata = json.loads((ROOT / filename).read_text())
        assert metadata["enable_gpu"] is False
        assert metadata["enable_internet"] is online
        assert not metadata.get("competition_sources")
        assert not metadata.get("machine_shape")


def _source_release(tmp_path, *, stale=False, unlisted=False, archived_manifest=False):
    """Real source code and metadata only; this fixture contains no readings."""
    root = tmp_path / "inputs" / "nested" / "source_release"
    for rel in GENERATOR.source_manifest()["files"]:
        destination = root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    if stale:
        (root / "src/awsad/__init__.py").write_text("raise AssertionError('stale source imported')\n")
    if archived_manifest:
        archived = root / "docs/research/prior_release/MANIFEST.json"
        archived.parent.mkdir(parents=True)
        archived.write_text(json.dumps({"schema_version": 1, "bundle_role": "online_builder_source",
                                       "processed_data_included": False,
                                       "training_data_policy": "real_observations_only", "sha256": {}}))
    manifest = {"schema_version": 1, "bundle_role": "online_builder_source",
                "processed_data_included": False, "training_data_policy": "real_observations_only",
                "sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in root.rglob("*") if p.is_file()}}
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    if unlisted:
        (root / "unlisted.txt").write_text("unlisted source-release file")
    return root


def _builder_bootstrap_check(tmp_path, *, expected_error=None):
    bootstrap = "".join(BUILDER.build_notebook()["cells"][2]["source"])
    bootstrap = bootstrap.replace('pathlib.Path("/kaggle/working")', 'pathlib.Path(' + repr(str(tmp_path / "working")) + ')')
    bootstrap = bootstrap.replace('pathlib.Path("/kaggle/input")', 'pathlib.Path(' + repr(str(tmp_path / "inputs")) + ')')
    script = "import builtins, sys\n" + """
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'pandas', 'numpy', 'requests', 'urllib'}:
        raise AssertionError('heavy/data/network import during source bootstrap: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
try:
"""
    script += "\n".join("    " + line for line in bootstrap.splitlines())
    script += "\nexcept Exception as exc:\n"
    if expected_error is None:
        script += "    raise\n"
    else:
        script += "    assert isinstance(exc, RuntimeError) and " + repr(expected_error) + " in str(exc), repr(exc)\n"
        script += "    assert 'awsad' not in sys.modules\n"
    script += "else:\n"
    script += ("    assert SOURCE_ROOT.is_dir() and OUT.is_dir()\n" if expected_error is None
               else "    raise AssertionError('invalid source release unexpectedly accepted')\n")
    script += "assert 'pandas' not in sys.modules and 'numpy' not in sys.modules\n"
    path = tmp_path / "builder_check.py"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("as_zip", [False, True])
def test_builder_accepts_nested_verified_source_before_data_imports(tmp_path, as_zip):
    release = _source_release(tmp_path)
    if as_zip:
        archive = tmp_path / "inputs" / "nested_source.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for path in release.rglob("*"):
                if path.is_file():
                    zf.write(path, "prefix/" + path.relative_to(release).as_posix())
        shutil.rmtree(release)  # Test-fixture path, not any project data.
    _builder_bootstrap_check(tmp_path)


@pytest.mark.parametrize("as_zip", [False, True])
def test_builder_ignores_archived_documentation_manifest_without_adjacent_source(tmp_path, as_zip):
    release = _source_release(tmp_path, archived_manifest=True)
    if as_zip:
        archive = tmp_path / "inputs" / "nested_source.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for path in release.rglob("*"):
                if path.is_file():
                    zf.write(path, "prefix/" + path.relative_to(release).as_posix())
        assert release.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(release)
    _builder_bootstrap_check(tmp_path)


@pytest.mark.parametrize("mode, error", [("stale", "Stale or missing source"),
                                         ("unlisted", "cover exactly"),
                                         ("zip_traversal", "unsafe source path")])
def test_builder_rejects_stale_unlisted_and_unsafe_source_before_import(tmp_path, mode, error):
    release = _source_release(tmp_path, stale=mode == "stale", unlisted=mode == "unlisted")
    if mode == "zip_traversal":
        archive = tmp_path / "inputs" / "nested_source.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for path in release.rglob("*"):
                if path.is_file():
                    zf.write(path, "prefix/" + path.relative_to(release).as_posix())
            zf.writestr("prefix/../escape.txt", "unsafe fixture member")
        shutil.rmtree(release)
    _builder_bootstrap_check(tmp_path, expected_error=error)


def test_generated_notebooks_match_current_generators_and_source():
    expected_builder = BUILDER.build_notebook()
    for filename in ("kaggle/aws_data_builder.ipynb", "kaggle_builder/aws_data_builder.ipynb"):
        assert json.loads((ROOT / filename).read_text(encoding="utf-8")) == expected_builder
    assert json.loads((ROOT / "kaggle/aws_anomaly_training.ipynb").read_text(encoding="utf-8")) == GENERATOR.build_notebook()
