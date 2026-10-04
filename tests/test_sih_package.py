"""Packaging tests use ONLY fake test fixtures, never submission observations."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import package_sih_release as package


def put(root, name, content=b"# Test-only fixture; no observation or credential.\n"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        content = package.encoded(content)
    if isinstance(content, str):
        content = content.encode()
    path.write_bytes(content)
    return path


def fixture_workspace(root, *, india=False, pattern_dir=package.PATTERN, selection_seal=False):
    for name in package.STATIC_FILES:
        put(root, name)
    for name in package.LICENSE_FILES:
        put(root, name, "License text fixture, not a redistribution claim.\n")
    put(root, "frontend/dist/assets/index-abc12345.js", "window.testFixture = true;")
    put(root, "frontend/dist/assets/index-abc12345.css", "body {color:black}")
    put(root, "frontend/dist/index.html", '<script src="/assets/index-abc12345.js"></script><link href="/assets/index-abc12345.css">')
    for directory in (package.MINUTE, package.REPLAY):
        for name in package.MINUTE_FILES:
            put(root, directory + "/" + name, {} if name.endswith(".json") else b"test-only artifact bytes")
        put(root, directory + "/detector.json", {"dataset_policy": "real_observations_only"})
        frozen_source = directory + "/source_snapshot/" + package.SOURCE_FILES[0]
        put(root, frozen_source, "# Test-only frozen source fixture.\n")
        shards = []
        for station in package.STATIONS:
            name = f"scored_observations/{station}_2025_01.parquet"
            source = put(root, directory + "/" + name, b"test-only fake parquet bytes")
            shards.append({"file": name, "sha256": package.sha256(source)})
        put(root, directory + "/provenance.json", {"source_files": {
            package.SOURCE_FILES[0]: package.sha256(root / frozen_source)}, "scored_shards": shards})
        put(root, directory + "/artifact_verification.json", {"status": "verified", **{
            name.split(".")[0] + "_sha256": package.sha256(root / directory / name)
            for name in ("models.joblib", "metrics.json", "provenance.json")}})
    for name in package.PATTERN_FILES:
        put(root, pattern_dir + "/" + name, {} if name.endswith(".json") else b"test-only synthetic-model fixture")
    freeze = {"policy": "synthetic_scenarios_on_original_observations",
              "model_sha256": package.sha256(root / pattern_dir / "pattern_model.joblib"),
              "isolation_baseline_sha256": package.sha256(root / pattern_dir / "isolation_baseline.joblib"),
              "source_fingerprints": {name: package.sha256(root / name)
                                      for name in package.SOURCE_SEAL_FILES},
              "frozen_minute_baseline": {name: package.sha256(root / package.MINUTE / name)
                                          for name in ("models.joblib", "detector.json", "config.json")},
              "environment": {"python": sys.version, "packages": {name: version(name) for name in package.RUNTIME_PACKAGES}}}
    if selection_seal:
        freeze["selection_script_sha256"] = package.sha256(root / "scripts/select_sih_model.py")
    frozen_path = put(root, pattern_dir + "/frozen.json", freeze)
    final = {"status": "final_test_complete", "frozen_sha256": package.sha256(frozen_path),
             "consumption_receipt": "artifacts_sih_test_registry/" + "a" * 64 + ".json",
             "fixture_notice": "TEST ONLY; no evaluation was performed"}
    final_path = put(root, pattern_dir + "/final_test.json", final)
    put(root, pattern_dir + "/metrics.json", {"status": "final_test_complete", "final_test": final})
    put(root, final["consumption_receipt"], {"status": "completed", "frozen_sha256": final["frozen_sha256"],
        "final_result_sha256": package.sha256(final_path), "fixture_notice": "TEST ONLY"})
    if india:
        station_files = []
        for station in package.INDIAN_STATIONS:
            name = f"baseline_observations/{station}_2024.parquet"
            path = put(root, package.INDIA + "/" + name, b"test-only Indian baseline fixture")
            station_files.append({"path": name, "sha256": package.sha256(path)})
        catalog = put(root, package.INDIA + "/station_catalog.csv", "station_id\n")
        put(root, package.INDIA + "/manifest.json", {"artifact_kind": "indian_station_demo", "eligible_for_real_training": False,
            "catalog_file": {"path": "station_catalog.csv", "sha256": package.sha256(catalog)}, "station_files": station_files})
    return root


def test_strict_allowlist_excludes_fake_tokens_environment_logs_and_unreferenced_assets(tmp_path, monkeypatch):
    root = fixture_workspace(tmp_path / "fixture", india=True)
    secrets = [".env", "kaggle-krishna.json", "kaggle/kaggle.json", "src/awsad/.env.local",
               "frontend/.env.production", "frontend/dist/.env", "frontend/dist/assets/unreferenced-abc12345.js",
               ".git/config", ".venv/secret.json", "data/old_corpus.parquet", "server.log"]
    for name in secrets:
        put(root, name, b"FAKE_SECRET_MUST_NEVER_ENTER_RELEASE")
    monkeypatch.setattr(Path, "rglob", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no recursive workspace scan")))
    plan = package.make_plan(root, include_data=True, include_india=True)
    assert plan.report()["ready"], plan.errors
    release = package.write_package(plan, tmp_path / "out")
    with zipfile.ZipFile(release["archive"]) as archive:
        names = set(archive.namelist())
        assert not names.intersection(secrets)
        assert "data/old_corpus.parquet" not in names
        assert f"{package.REPLAY}/scored_observations/bon_2025_01.parquet" in names
        assert f"{package.MINUTE}/scored_observations/bon_2025_01.parquet" not in names
        assert all(b"FAKE_SECRET_MUST_NEVER_ENTER_RELEASE" not in archive.read(name) for name in names)
        inventory = json.loads(archive.read("release_manifest.json"))
        assert set(inventory["files"]) == names - {"release_manifest.json"}
        for name, entry in inventory["files"].items():
            assert package.hashlib.sha256(archive.read(name)).hexdigest() == entry["sha256"]
        assert "third_party/licenses/react-LICENSE" in names
        settings = json.loads(archive.read("release_settings.json"))
        assert settings["replay_directory"] == package.REPLAY
        assert settings["pattern_directory"] == package.PATTERN


def test_identical_inputs_produce_identical_zip_and_existing_release_is_preserved(tmp_path):
    root = fixture_workspace(tmp_path / "fixture")
    plan = package.make_plan(root, include_data=True)
    first = package.write_package(plan, tmp_path / "one")
    second = package.write_package(plan, tmp_path / "two")
    assert first["sha256"] == second["sha256"]
    assert Path(first["archive"]).read_bytes() == Path(second["archive"]).read_bytes()
    with pytest.raises(FileExistsError):
        package.write_package(plan, tmp_path / "one")
    assert package.sha256(first["archive"]) == first["sha256"]


@pytest.mark.parametrize("problem", ["missing_model", "unfinished_evaluation", "changed_model", "missing_model_hash", "stale_build", "missing_asset"])
def test_incomplete_or_changed_inputs_block_packaging(tmp_path, problem):
    root = fixture_workspace(tmp_path / "fixture")
    if problem == "missing_model":
        (root / package.PATTERN / "pattern_model.joblib").unlink()
    elif problem == "unfinished_evaluation":
        put(root, package.PATTERN + "/metrics.json", {"status": "frozen_awaiting_final_test"})
    elif problem == "changed_model":
        put(root, package.PATTERN + "/pattern_model.joblib", b"changed test-only bytes")
    elif problem == "missing_model_hash":
        path = root / package.PATTERN / "frozen.json"
        value = json.loads(path.read_text())
        value.pop("model_sha256")
        put(root, package.PATTERN + "/frozen.json", value)
    elif problem == "stale_build":
        os.utime(root / "frontend/src/App.tsx", (2_000_000_000, 2_000_000_000))
    else:
        (root / "frontend/dist/assets/index-abc12345.js").unlink()
    plan = package.make_plan(root)
    assert not plan.report()["ready"]
    with pytest.raises(ValueError, match="not ready"):
        package.write_package(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_source_changes_after_check_leave_no_incomplete_zip(tmp_path):
    root = fixture_workspace(tmp_path / "fixture")
    plan = package.make_plan(root)
    put(root, "README.md", b"changed after check")
    with pytest.raises(ValueError, match="changed after release check"):
        package.write_package(plan, tmp_path / "out")
    assert not list((tmp_path / "out").iterdir())


@pytest.mark.parametrize("name", ["../.env", "C:/secret.json", "src/../../kaggle.json", "frontend/.env.local", ".git/config"])
def test_unsafe_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        package.safe_file(tmp_path, name)


def test_symlink_cannot_smuggle_external_content(tmp_path):
    root = fixture_workspace(tmp_path / "fixture")
    target = put(tmp_path, "outside.txt", b"FAKE_EXTERNAL_SECRET")
    (root / "README.md").unlink()
    try:
        (root / "README.md").symlink_to(target)
    except OSError:
        pytest.skip("OS does not permit test symlinks")
    assert any("symlink" in error for error in package.make_plan(root).errors)


@pytest.mark.parametrize("method", ["is_symlink", "is_junction"])
def test_link_guard_without_os_symlink_privileges(tmp_path, monkeypatch, method):
    root = fixture_workspace(tmp_path / "fixture")
    original = getattr(Path, method, lambda path: False)
    monkeypatch.setattr(Path, method, lambda path: path.name == "README.md" or original(path), raising=False)
    assert any("symlink/junction" in error for error in package.make_plan(root).errors)


def test_data_omission_is_explicit_and_launcher_verifies_inventory(tmp_path):
    root = fixture_workspace(tmp_path / "fixture")
    omitted = package.make_plan(root)
    assert omitted.report()["ready"] and not omitted.report()["dashboard_ready_with_bundled_data"]
    assert omitted.warnings and not any("scored_observations/" in name for name in omitted.files)
    complete = package.make_plan(root, include_data=True)
    release = package.write_package(complete, tmp_path / "out")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(release["archive"]) as archive:
        archive.extractall(extracted)
    checked = subprocess.run([sys.executable, str(extracted / "launch_sih.py"), "--check"], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["inventory_verified"]
    (extracted / "README.md").write_text("corruption")
    corrupted = subprocess.run([sys.executable, str(extracted / "launch_sih.py"), "--check"], capture_output=True, text=True)
    assert corrupted.returncode != 0 and "hash mismatch" in corrupted.stderr


def test_selected_artifact_directory_maps_to_stable_archive_path_without_fallback(tmp_path):
    selected = "artifacts_sih_tree_selected_20260930"
    root = fixture_workspace(tmp_path / "fixture", pattern_dir=selected, selection_seal=True)
    put(root, package.PATTERN + "/pattern_model.joblib", b"UNSELECTED_DECOY_MUST_NOT_BE_USED")
    plan = package.make_plan(root, pattern_dir=selected, include_data=True)
    assert not plan.errors, plan.errors
    release = package.write_package(plan, tmp_path / "out")
    with zipfile.ZipFile(release["archive"]) as archive:
        names = archive.namelist()
        assert not any(name.startswith(selected + "/") for name in names)
        assert archive.read(package.PATTERN + "/pattern_model.joblib") == b"test-only synthetic-model fixture"
        inventory = json.loads(archive.read("release_manifest.json"))
        assert inventory["files"][package.PATTERN + "/pattern_model.joblib"]["source"] == selected + "/pattern_model.joblib"
        settings = json.loads(archive.read("release_settings.json"))
        assert settings["pattern_directory"] == package.PATTERN
        assert settings["options"]["pattern_dir"] == selected
        for name in ("scripts/select_sih_model.py", "tests/test_pattern_heads.py", "tests/test_sih_runtime_review.py"):
            assert name in names
        extracted = tmp_path / "selected_release"
        archive.extractall(extracted)
    checked = subprocess.run([sys.executable, str(extracted / "launch_sih.py"), "--check"], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr


@pytest.mark.parametrize("name", ["scripts/run_minute_detection.py", "scripts/select_sih_model.py"])
def test_helper_and_optional_selection_seals_are_enforced(tmp_path, name):
    root = fixture_workspace(tmp_path / "fixture", selection_seal=True)
    put(root, name, b"changed after frozen protocol")
    assert "hash mismatch: " + name in package.make_plan(root).errors


def test_missing_minute_helper_in_source_set_is_rejected(tmp_path):
    root = fixture_workspace(tmp_path / "fixture")
    path = root / package.PATTERN / "frozen.json"
    freeze = json.loads(path.read_text())
    freeze["source_fingerprints"].pop("scripts/run_minute_detection.py")
    put(root, package.PATTERN + "/frozen.json", freeze)
    assert any("source set differs" in error for error in package.make_plan(root).errors)


@pytest.mark.parametrize("selected", ["", ".", "..", "../artifacts_other", "/tmp/artifacts_other", "C:/private/artifacts", "data/.env", ".git/models"])
def test_pattern_directory_rejects_unsafe_paths_before_reading_files(tmp_path, monkeypatch, selected):
    def unexpected_hash(path):
        raise AssertionError("unsafe pattern directory reached a file read")
    monkeypatch.setattr(package, "sha256", unexpected_hash)
    plan = package.make_plan(tmp_path, pattern_dir=selected)
    assert plan.errors and not plan.files


def test_pattern_directory_rejects_symlink_even_when_target_would_be_inside_workspace(tmp_path, monkeypatch):
    selected = "artifacts_sih_selected"
    root = fixture_workspace(tmp_path / "fixture", pattern_dir=selected)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.name == selected or original(path))
    plan = package.make_plan(root, pattern_dir=selected)
    assert any("symlink" in error for error in plan.errors)
    assert not plan.files


def test_check_cli_accepts_selected_pattern_directory_without_creating_zip(tmp_path):
    selected = "artifacts_sih_selected"
    root = fixture_workspace(tmp_path / "fixture", pattern_dir=selected, selection_seal=True)
    checked = subprocess.run([sys.executable, package.__file__, "--workspace", str(root), "--pattern-dir", selected,
                              "--include-data", "--check"], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr + checked.stdout
    report = json.loads(checked.stdout)
    assert report["ready"] and report["options"]["pattern_dir"] == selected
    assert not report["archive_created"] and not (root / "out_sih_release").exists()
