"""Offline notebook provenance tests; fixtures contain no training observations."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("minute_notebook_generator", ROOT / "kaggle/build_minute_notebook.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


@pytest.fixture
def helpers():
    namespace = {}
    exec(GENERATOR.BOOTSTRAP_HELPERS, namespace)
    return namespace


def metadata_bundle(path):
    """Integrity-only dummy release; never eligible for native raw admission."""
    path.mkdir(parents=True)
    files = {"data/processed/source_metadata.json": b'{"software_test_fixture": true}',
             "provenance/source_manifest.json": b'{"role": "metadata_test"}',
             "src/awsad/__init__.py": b'raise AssertionError("archived source must not execute")\n'}
    for rel, raw in files.items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = {"schema_version": 1, "bundle_role": "processed_training_bundle",
                "source": "noaa_surfrad", "training_data_policy": "real_observations_only",
                "raw_observations_retained": True, "unknown_fault_labels": True,
                "source_manifest_sha256": hashlib.sha256(files["provenance/source_manifest.json"]).hexdigest(),
                "training_source_sha256": "archived-hourly-code-not-minute-code",
                "build_id": "metadata-fixture-only",
                "sha256": {rel: hashlib.sha256(raw).hexdigest() for rel, raw in files.items()}}
    (path / "MANIFEST.json").write_text(json.dumps(manifest))
    return manifest


def zip_bundle(root, target, *, prefix="bundle/"):
    with zipfile.ZipFile(target, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, prefix + path.relative_to(root).as_posix())


def test_embedded_source_is_exact_transitive_minute_code_and_cells_compile(tmp_path, helpers):
    manifest, payload = GENERATOR.embedded_source()
    materialized = helpers["materialize_source"](manifest, payload, tmp_path / "source")
    assert manifest["role"] == "embedded_native_minute_source"
    assert set(manifest["files"]) == {
        "scripts/run_minute_detection.py", "src/awsad/__init__.py",
        "src/awsad/minute_detection.py", "src/awsad/data/surfrad.py",
        "src/awsad/data/surfrad_native.py", "src/awsad/data/acquisition.py",
        "src/awsad/data/fetch_surfrad.py", "src/awsad/data/verify_surfrad.py",
        "src/awsad/evaluation/real_events.py"}
    for rel in manifest["files"]:
        assert (materialized / rel).read_bytes() == (ROOT / rel).read_bytes()
    notebook = GENERATOR.build_notebook()
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for code in cells:
        compile(code, "minute_notebook_cell", "exec")
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(cells[0]).body}
    assert constants["CODE_MANIFEST"] == manifest
    assert constants["MINUTE_CONFIG"]["calibration_end"] == "2024-11-01T00:00:00Z"
    assert constants["MINUTE_CONFIG"]["holdout_station_ids"] == ["gwn"]
    # Prove the isolated snapshot can start without the repository on sys.path.
    result = subprocess.run([sys.executable, "-B", str(materialized / "scripts/run_minute_detection.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--replay-only" in result.stdout


def test_tampered_embedded_code_is_rejected_before_materialization(tmp_path, helpers):
    manifest, payload = GENERATOR.embedded_source()
    contents = json.loads(zlib.decompress(base64.b64decode(payload)))
    contents["src/awsad/minute_detection.py"] = base64.b64encode(b"raise RuntimeError('tampered')").decode()
    broken = base64.b64encode(zlib.compress(json.dumps(contents).encode())).decode()
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        helpers["materialize_source"](manifest, broken, tmp_path / "bad-source")
    assert not (tmp_path / "bad-source").exists()


def test_retained_observation_chain_does_not_execute_archived_hourly_code(tmp_path, helpers):
    root = tmp_path / "inputs/processed"
    manifest = metadata_bundle(root)
    chosen = helpers["locate_bundles"](tmp_path / "inputs")
    assert chosen["path"] == root
    report = helpers["verify_processed_bundle"](root, manifest)
    assert report["eligible"]
    assert report["original_hourly_training_source_sha256"] == "archived-hourly-code-not-minute-code"
    assert "native measurement admission follows" in report["role"]


@pytest.mark.parametrize("change", ["tampered", "extra", "missing_provenance", "legacy_policy"])
def test_reject_changed_or_incomplete_processed_release(tmp_path, helpers, change):
    root = tmp_path / "processed"
    manifest = metadata_bundle(root)
    if change == "tampered":
        (root / "data/processed/source_metadata.json").write_text("changed")
    elif change == "extra":
        (root / "unexpected.txt").write_text("not manifest-listed")
    elif change == "missing_provenance":
        manifest["source_manifest_sha256"] = "0" * 64
    else:
        manifest["training_data_policy"] = "legacy_injected"
    with pytest.raises(RuntimeError):
        helpers["verify_processed_bundle"](root, manifest)


def test_unpacked_and_zip_same_release_allowed_but_distinct_releases_rejected(tmp_path, helpers):
    inputs = tmp_path / "inputs"
    root = inputs / "processed"
    metadata_bundle(root)
    zip_bundle(root, inputs / "duplicate.zip")
    assert helpers["locate_bundles"](inputs)["kind"] == "root"
    other = inputs / "different"
    manifest = metadata_bundle(other)
    manifest["build_id"] = "another-build"
    (other / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="ambiguous"):
        helpers["locate_bundles"](inputs)


def test_zip_only_release_materializes_and_verifies(tmp_path, helpers):
    source = tmp_path / "source"
    metadata_bundle(source)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    zip_bundle(source, inputs / "bundle.zip")
    chosen = helpers["locate_bundles"](inputs)
    root = helpers["materialize_bundle"](chosen, tmp_path / "extracted")
    assert helpers["verify_processed_bundle"](root, chosen["manifest"])["checked_files"] == 3


@pytest.mark.parametrize("bad_name", ["../escape.py", "C:/escape.py", "nested\\escape.py"])
def test_zip_path_escape_is_rejected(tmp_path, helpers, bad_name):
    source = tmp_path / "source"
    metadata_bundle(source)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    target = inputs / "bundle.zip"
    zip_bundle(source, target)
    with zipfile.ZipFile(target, "a") as archive:
        archive.writestr("bundle/" + bad_name, "untrusted")
    if "\\" in bad_name:
        # Windows ZipInfo normalizes separators when writing. Recreate the
        # actual foreign ZIP spelling in both headers without changing size.
        target.write_bytes(target.read_bytes().replace(
            ("bundle/" + bad_name.replace("\\", "/")).encode(),
            ("bundle/" + bad_name).encode()))
    chosen = helpers["locate_bundles"](inputs)
    with pytest.raises(RuntimeError, match="unsafe bundle path"):
        helpers["materialize_bundle"](chosen, tmp_path / "extracted")
    assert not (tmp_path / "escape.py").exists()


def test_zip_duplicate_members_rejected(tmp_path, helpers):
    source = tmp_path / "source"
    metadata_bundle(source)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    target = inputs / "bundle.zip"
    zip_bundle(source, target)
    with zipfile.ZipFile(target, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr("bundle/provenance/source_manifest.json", "{}")
    with pytest.raises(RuntimeError, match="duplicate bundle"):
        helpers["materialize_bundle"](helpers["locate_bundles"](inputs), tmp_path / "extracted")


def test_committed_notebook_is_current_and_offline_metadata_matches():
    expected = GENERATOR.build_notebook()
    for relative in ("kaggle/aws_minute_detection.ipynb", "kaggle_minute/aws_minute_detection.ipynb"):
        assert json.loads((ROOT / relative).read_text(encoding="utf-8")) == expected
    metadata = json.loads((ROOT / "kaggle_minute/kernel-metadata.json").read_text())
    assert (ROOT / "kaggle_minute" / metadata["code_file"]).is_file()
    assert metadata["enable_gpu"] is metadata["enable_tpu"] is metadata["enable_internet"] is False
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["krishnagupta02468/skyguard-sih26073-processed"]
