"""Synthetic software fixtures only; never dataset observations or labels."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from awsad.evaluation.real_events import review_template

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reviewed_event_cli", ROOT / "scripts/evaluate_real_events.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def artifact_fixture(tmp_path):
    artifacts = tmp_path / "test-only-artifacts"
    artifacts.mkdir()
    rows = pd.DataFrame({
        "group": "TEST_ONLY|fixture", "timestamp": pd.date_range("2025-01-31T23:58:00Z", periods=4, freq="min"),
        "observation_id": [f"TEST_ONLY_{i}" for i in range(4)],
        "split": "test_only", "threshold_group": "TEST_ONLY|fixture",
    })
    for channel in CLI.CHANNELS:
        rows[channel + "__alert"] = pd.array([False, True, True, None] if channel == "temperature_c"
                                                else [False, False, False, False], dtype="boolean")
        rows[channel + "__abrupt_change"] = [0.0, 3.0, 3.0, float("nan")]
    shards = []
    for i, part in enumerate((rows.iloc[:2], rows.iloc[2:])):
        path = artifacts / f"test_only_{i}.parquet"
        part.to_parquet(path, index=False)
        shards.append({"file": path.name, "sha256": CLI.sha256(path), "rows": len(part),
                       "group": "TEST_ONLY|fixture", "start": part.timestamp.iloc[0].isoformat(),
                       "end": part.timestamp.iloc[-1].isoformat()})
    (artifacts / "provenance.json").write_text(json.dumps({"scored_shards": shards}))
    (artifacts / "detector.json").write_text(json.dumps({
        "thresholds": {"TEST_ONLY|fixture": {channel: {"abrupt_change": {"threshold": 2.0,
                          "comparison": "strictly_greater"}} for channel in CLI.CHANNELS}}}))
    reviews = tmp_path / "test_only_reviews.csv"
    review_template().to_csv(reviews, index=False)
    return artifacts, reviews


def review_fixture(status="reviewed_background", **changes):
    result = {
        "review_id": "TEST_ONLY_REVIEW", "event_id": "TEST_ONLY_EVENT", "group": "TEST_ONLY|fixture",
        "channel": "temperature_c", "start_utc": "2025-01-31T23:58:00Z", "end_utc": "2025-02-01T00:02:00Z",
        "review_status": status, "onset_earliest_utc": "2025-01-31T23:57:00Z",
        "onset_latest_utc": "2025-01-31T23:58:00Z", "reviewer": "TEST FIXTURE ONLY",
        "reviewed_at_utc": "2025-02-02T00:00:00Z", "evidence_basis": "operator_log",
        "evidence_refs": json.dumps([{"url": "https://example.invalid/test-fixture", "sha256": "a" * 64}]),
        "review_notes": "Test only. Never use as observed or ground-truth data.",
    }
    result.update(changes)
    return result


def test_empty_review_preserves_unknown_counts_and_missing_decisions(tmp_path):
    artifacts, reviews = artifact_fixture(tmp_path)
    report = CLI.evaluate_artifacts(artifacts, reviews)
    assert report["status"] == "unverified_no_reviewed_truth"
    assert report["input_native_observation_rows"] == 4
    assert report["input_observation_rows"] == report["unknown_observation_rows"] == 12
    assert report["unknown_alert_rows"] == 2
    assert report["event_recall"] is None
    assert report["false_alert_fraction"] is None
    assert report["inputs"]["reviews_sha256"] == CLI.sha256(reviews)
    assert len(report["inputs"]["scored_shards"]) == 2
    json.dumps(report, allow_nan=False)


def test_reviewed_episode_and_exposure_cross_monthly_shards(tmp_path):
    artifacts, reviews = artifact_fixture(tmp_path)
    pd.DataFrame([review_fixture()]).to_csv(reviews, index=False)
    report = CLI.evaluate_artifacts(artifacts, reviews)
    assert report["reviewed_observation_rows"] == 4
    assert report["unknown_observation_rows"] == 8
    assert report["unknown_alert_rows"] == 0
    assert report["false_alert_rows"] == 2
    assert report["false_alert_episodes"] == 1
    assert report["background_scored_rows"] == 3  # a null decision is not a negative
    assert report["background_contiguous_scored_seconds"] == 120


def test_independent_event_delay_and_saved_baseline_comparison(tmp_path):
    artifacts, reviews = artifact_fixture(tmp_path)
    pd.DataFrame([review_fixture("confirmed_event")]).to_csv(reviews, index=False)
    report = CLI.evaluate_artifacts(artifacts, reviews, signal="abrupt_change")
    assert report["event_recall"] == 1.0
    assert report["mean_detection_delay_seconds_min"] == 60
    assert report["mean_detection_delay_seconds_max"] == 120
    assert report["detector_decision"] == "saved_signal_abrupt_change"
    assert report["thresholds_retuned"] is False
    assert report["unknown_alert_rows"] == 4  # other channels remain unknown


def test_provider_qc_cannot_be_promoted_to_reviewed_truth(tmp_path):
    artifacts, reviews = artifact_fixture(tmp_path)
    pd.DataFrame([review_fixture(evidence_basis="provider_qc")]).to_csv(reviews, index=False)
    with pytest.raises(ValueError, match="independent evidence"):
        CLI.evaluate_artifacts(artifacts, reviews)


def test_changed_shard_and_duplicate_manifest_paths_are_rejected(tmp_path):
    artifacts, reviews = artifact_fixture(tmp_path)
    provenance_path = artifacts / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["scored_shards"].append(provenance["scored_shards"][0])
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="unique"):
        CLI.evaluate_artifacts(artifacts, reviews)
    provenance["scored_shards"].pop()
    provenance_path.write_text(json.dumps(provenance))
    (artifacts / "test_only_0.parquet").write_bytes(b"TEST ONLY CHANGED FILE")
    with pytest.raises(ValueError, match="bytes changed"):
        CLI.evaluate_artifacts(artifacts, reviews)


def test_overlapping_shards_are_rejected_even_if_individual_hashes_match(tmp_path):
    artifacts, reviews = artifact_fixture(tmp_path)
    provenance_path = artifacts / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    (artifacts / "test_only_1.parquet").write_bytes((artifacts / "test_only_0.parquet").read_bytes())
    provenance["scored_shards"][1] = dict(provenance["scored_shards"][0], file="test_only_1.parquet")
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="overlap"):
        CLI.evaluate_artifacts(artifacts, reviews)
