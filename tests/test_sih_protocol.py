"""Protocol probes use temporary software fixtures only, never held-out data."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_sih_benchmark.py"
    spec = importlib.util.spec_from_file_location("sih_protocol_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scope(stations=("bon",), start="2038-02-01T00:00:00Z", end="2038-03-01T00:00:00Z"):
    return {"source_product": "SOFTWARE_FIXTURE_ONLY", "stations": list(stations),
            "start": start, "end_exclusive": end}


def test_registry_blocks_different_models_output_directories_and_overlapping_subsets(runner, tmp_path):
    registry = tmp_path / "registry"
    receipt = runner.claim_final_test(scope(("bon", "gwn")), "frozen-a", "manifest-a", tmp_path / "run-a", registry=registry)
    runner.finish_final_test(receipt, "frozen-a", "failed", error="software fixture interruption")
    with pytest.raises(FileExistsError, match="already consumed"):
        runner.claim_final_test(scope(("bon", "gwn")), "frozen-b", "manifest-a", tmp_path / "run-b", registry=registry)
    with pytest.raises(FileExistsError, match="already consumed"):
        runner.claim_final_test(scope(("bon",)), "frozen-a", "manifest-a", tmp_path / "run-c", registry=registry)
    with pytest.raises(FileExistsError, match="already consumed"):
        runner.claim_final_test(scope(("bon", "gwn")), "frozen-a", "changed-manifest", tmp_path / "run-a", registry=registry)


def test_same_frozen_failed_attempt_can_resume_but_completed_or_started_cannot(runner, tmp_path):
    registry = tmp_path / "registry"
    receipt = runner.claim_final_test(scope(), "frozen-a", "manifest-a", tmp_path / "run", registry=registry)
    with pytest.raises(RuntimeError, match="still marked started"):
        runner.claim_final_test(scope(), "frozen-a", "manifest-a", tmp_path / "run", registry=registry)
    runner.finish_final_test(receipt, "frozen-a", "failed", error="fixture failure")
    same = runner.claim_final_test(scope(), "frozen-a", "manifest-a", tmp_path / "run", registry=registry)
    assert same == receipt
    result = tmp_path / "fixture_result.json"
    result.write_text('{"software_fixture": true}')
    runner.finish_final_test(receipt, "frozen-a", "completed", final_path=result)
    data = json.loads(receipt.read_text())
    assert data["status"] == "completed"
    assert len(data["attempts"]) == 2
    assert data["final_result_sha256"] == runner.sha256_file(result)
    with pytest.raises(FileExistsError, match="already consumed"):
        runner.claim_final_test(scope(), "frozen-a", "manifest-a", tmp_path / "copied-run", registry=registry)
    # A genuinely different period is not blocked by a prior receipt.
    other = runner.claim_final_test(scope(start="2038-03-01T00:00:00Z", end="2038-04-01T00:00:00Z"),
                                    "frozen-b", "manifest-b", tmp_path / "next", registry=registry)
    assert other != receipt


def test_interval_audit_detects_overlapping_histories_with_distinct_window_ids(runner):
    def window(name, start, end, station="bon"):
        return {"window_id": name, "station_id": station, "start": start, "end": end}
    first = window("first", "2038-01-01T00:00:00Z", "2038-01-01T11:59:00Z")
    overlap = window("different-id", "2038-01-01T11:00:00Z", "2038-01-01T22:59:00Z")
    partitions = {"fit": (None, {"source_windows": [first]}),
                  "selection": (None, {"source_windows": [overlap]})}
    with pytest.raises(ValueError, match="overlapping source histories"):
        runner.audit_partitions(partitions)
    overlap["start"] = "2038-01-01T12:00:00Z"
    assert runner.audit_partitions(partitions)["overlapping_source_windows"] == 0


def test_cache_reuse_rejects_changed_scenario_files_and_changed_arrays(runner, tmp_path, monkeypatch):
    cache, destination = tmp_path / "source", tmp_path / "scenarios"
    cache.mkdir()
    destination.mkdir()
    (cache / "native_manifest.json").write_text('{"fixture": true}')
    copies = destination / "fit"
    copies.mkdir()
    artifact = copies / "00000_no_injection.parquet"
    artifact.write_bytes(b"hash-only software fixture, never parsed as parquet")
    dataset = {"X": np.zeros((1, 1), np.float32), "y": np.array(["no_injection"]),
               "scenario_indices": np.array([0], np.int32), "positions": np.array([180], np.int16)}
    arrays = destination / "fit.npz"
    np.savez_compressed(arrays, **dataset)
    metadata = {"spec": {"cache_schema_version": 2, "version": runner.VERSION, "feature_version": runner.FEATURE_VERSION,
        "split": "fit", "windows_per_shard": 1, "seed": 1,
        "source_manifest_sha256": runner.sha256_file(cache / "native_manifest.json"),
        "generator_sha256": runner.sha256_file(runner.ROOT / "src/awsad/benchmark/scenarios.py"),
        "features_sha256": runner.sha256_file(runner.ROOT / "src/awsad/benchmark/operational_model.py")},
        "arrays_sha256": runner.sha256_file(arrays), "rows": 1, "features": ["fixture"],
        "events": [{"artifact": "fit/00000_no_injection.parquet", "artifact_sha256": runner.sha256_file(artifact)}]}
    runner.write(destination / "fit.json", metadata)
    monkeypatch.setattr(runner, "source_windows", lambda *a, **k: pytest.fail("cache reuse must not parse sources"))
    restored, _ = runner.make_partition(cache, destination, "fit", 1, 1)
    np.testing.assert_array_equal(restored["y"], dataset["y"])
    artifact.write_bytes(b"changed scenario")
    with pytest.raises(ValueError, match="scenario copy hash/path mismatch"):
        runner.make_partition(cache, destination, "fit", 1, 1)
    artifact.write_bytes(b"hash-only software fixture, never parsed as parquet")
    arrays.write_bytes(b"changed arrays")
    with pytest.raises(ValueError, match="cached arrays changed"):
        runner.make_partition(cache, destination, "fit", 1, 1)


def test_source_fingerprints_cover_existing_baseline_implementation(runner):
    fingerprints = runner.source_fingerprints()
    assert "src/awsad/minute_detection.py" in fingerprints
    assert "src/awsad/benchmark/injection_eval.py" in fingerprints
    assert "src/awsad/data/acquisition.py" in fingerprints
    assert "scripts/run_minute_detection.py" in fingerprints


def native_manifest(acquisition_hash, test_scope):
    return {"evidence": {"manifest_path": "surfrad_acquisition.json", "manifest_sha256": acquisition_hash,
                "index_verified": True, "evidence_fingerprint": "fixture-evidence"},
            "identity": {"source_fingerprint": "fixture-evidence", "start": test_scope["start"], "end": test_scope["end_exclusive"]},
            "scope": {"stations": test_scope["stations"]}, "raw_rows_reconstructed": True, "synthetic_observations": 0}


def test_native_cache_binding_rejects_unlinked_existing_cache(runner, tmp_path):
    registry, cache = tmp_path / "registry", tmp_path / "cache"
    cache.mkdir()
    receipt = runner.claim_final_test(scope(), "frozen-a", "acquisition-a", tmp_path / "out", registry=registry)
    manifest = native_manifest("different-acquisition", scope())
    runner.write(cache / "native_manifest.json", manifest)
    with pytest.raises(ValueError, match="does not match the claimed raw acquisition"):
        runner.bind_native_test_cache(receipt, "frozen-a", "acquisition-a", cache, manifest, scope())
    manifest = native_manifest("acquisition-a", scope())
    runner.write(cache / "native_manifest.json", manifest)
    native_hash = runner.bind_native_test_cache(receipt, "frozen-a", "acquisition-a", cache, manifest, scope())
    stored = json.loads(receipt.read_text())
    assert stored["test_acquisition_manifest_sha256"] == "acquisition-a"
    assert stored["native_cache_manifest_sha256"] == native_hash
    manifest["created_at_utc"] = "fixture-changed-cache"
    runner.write(cache / "native_manifest.json", manifest)
    with pytest.raises(ValueError, match="changed after an earlier attempt"):
        runner.bind_native_test_cache(receipt, "frozen-a", "acquisition-a", cache, manifest, scope())


def test_evaluation_claims_consumption_before_native_cache_construction(runner, tmp_path, monkeypatch):
    output, scenarios, test_cache, baseline, archive = [tmp_path / name for name in
                                                       ("run", "scenarios", "test-cache", "baseline", "raw-archive")]
    for directory in (output, scenarios, baseline, archive):
        directory.mkdir()
    (output / "pattern_model.joblib").write_bytes(b"software-only model fixture")
    (output / "isolation_baseline.joblib").write_bytes(b"software-only baseline fixture")
    acquisition_path = archive / "surfrad_acquisition.json"
    acquisition_path.write_text('{"software_fixture": true}')
    for name in ("models.joblib", "detector.json", "config.json"):
        (baseline / name).write_bytes(b"software fixture")
    development_hashes = {}
    for split in ("fit", "selection", "calibration"):
        arrays = scenarios / f"{split}.npz"
        arrays.write_bytes(b"hash-only development fixture")
        metadata = {"spec": {"split": split}, "arrays_sha256": runner.sha256_file(arrays), "source_windows": []}
        runner.write(scenarios / f"{split}.json", metadata)
        development_hashes[split] = runner.sha256_file(scenarios / f"{split}.json")
    monkeypatch.setattr(runner, "source_fingerprints", lambda: {"fixture": "source-a"})
    monkeypatch.setattr(runner, "runtime_versions", lambda: {"fixture": "1"})
    monkeypatch.setattr(runner, "FINAL_TEST_SCOPE", scope())
    monkeypatch.setattr(runner, "TEST_REGISTRY", tmp_path / "registry")
    monkeypatch.setattr(runner.OperationalPatternModel, "load", lambda p: object())
    monkeypatch.setattr(runner.joblib, "load", lambda p: object())
    freeze = {"source_fingerprints": {"fixture": "source-a"},
        "model_sha256": runner.sha256_file(output / "pattern_model.joblib"),
        "isolation_baseline_sha256": runner.sha256_file(output / "isolation_baseline.joblib"),
        "environment": {"python": runner.sys.version, "packages": {"fixture": "1"}},
        "final_test_scope": scope(), "scenario_metadata_sha256": development_hashes,
        "frozen_minute_baseline": {name: runner.sha256_file(baseline / name) for name in ("models.joblib", "detector.json", "config.json")},
        "final_windows_per_shard": 1, "seed": 1}
    runner.write(output / "frozen.json", freeze)
    def fake_cache_builder(archive_arg, cache_arg, scope_arg):
        assert Path(archive_arg) == archive
        assert not test_cache.exists()
        records = list(runner.TEST_REGISTRY.glob("*.json"))
        assert len(records) == 1
        receipt = json.loads(records[0].read_text())
        assert receipt["status"] == "started"
        assert receipt["test_acquisition_manifest_sha256"] == runner.sha256_file(acquisition_path)
        assert "native_cache_manifest_sha256" not in receipt
        test_cache.mkdir()
        manifest = native_manifest(runner.sha256_file(acquisition_path), scope_arg)
        runner.write(test_cache / "native_manifest.json", manifest)
        return manifest
    def stop_before_data(*args, **kwargs):
        receipt = json.loads(next(runner.TEST_REGISTRY.glob("*.json")).read_text())
        assert receipt["native_cache_manifest_sha256"] == runner.sha256_file(test_cache / "native_manifest.json")
        raise RuntimeError("synthetic stop before final data parsing")
    monkeypatch.setattr(runner, "build_native_test_cache", fake_cache_builder)
    monkeypatch.setattr(runner, "make_partition", stop_before_data)
    args = SimpleNamespace(out=str(output), scenarios=str(scenarios), test_cache=str(test_cache),
                           test_archive=str(archive), minute_artifacts=str(baseline))
    with pytest.raises(RuntimeError, match="synthetic stop"):
        runner.evaluate_final(args)
    receipt = json.loads(next(runner.TEST_REGISTRY.glob("*.json")).read_text())
    assert receipt["status"] == "failed"
    assert receipt["test_consumed_do_not_retune"] is True
    assert not (output / "final_test.json").exists()
