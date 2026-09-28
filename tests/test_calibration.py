"""Software regression fixtures only; these are not weather observations."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from awsad.evaluation.calibration import temporal_partitions, balanced_group_class_weights
from awsad.evaluation.metrics import choose_threshold, pointwise
from awsad.train_pipeline import _binary_report


@pytest.mark.parametrize("seed", range(10))
def test_exact_threshold_matches_brute_force_including_ties(seed):
    rng = np.random.default_rng(seed)
    scores = np.round(rng.normal(size=120), 1)
    y = rng.integers(0, 2, 120).astype(float)
    y[::7] = np.nan
    scores[::13] = np.nan
    valid = np.isfinite(scores) & np.isin(y, [0, 1])
    candidates = np.r_[np.nextafter(scores[valid].max(), np.inf),
                       np.sort(np.unique(scores[valid]))[::-1]]
    objectives = []
    for threshold in candidates:
        metric = pointwise(y[valid], scores[valid] >= threshold)
        objectives.append(metric["f1"] * (.5 if metric["precision"] < .35 else 1))
    threshold, detail = choose_threshold(scores, y)
    assert threshold == candidates[np.argmax(objectives)]
    assert detail["objective"] == pytest.approx(max(objectives))
    assert detail["n_labeled"] == valid.sum()


def test_constant_negative_calibration_can_choose_no_alarms():
    scores = np.zeros(200)
    threshold, detail = choose_threshold(scores, np.zeros(200))
    assert threshold > 0 and not np.any(scores >= threshold)
    assert detail["no_alarm"]


def test_constant_train_scores_do_not_trigger_all_pot_fallback_alarms():
    from awsad.evaluation.thresholds import pot_threshold
    for n in (50, 2000):
        scores = np.zeros(n)
        assert pot_threshold(scores) > 0
    with pytest.raises(ValueError, match="finite"):
        pot_threshold(np.array([np.nan]))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_no_alarm_threshold_survives_serving_score_precision(dtype):
    from awsad.evaluation.thresholds import pot_threshold
    for value in (0, .2, -1):
        scores = np.full(200, value, dtype=dtype)
        threshold, detail = choose_threshold(scores, np.zeros(200))
        assert detail["no_alarm"] and not np.any(scores >= threshold)
        assert not np.any(scores >= pot_threshold(scores))


def test_event_report_keeps_sources_separate_and_inclusive_endpoints():
    from awsad.train_pipeline import per_fault_breakdown
    ts = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame({"station_id": ["x"] * 8,
                          "source": ["a"] * 4 + ["b"] * 4,
                          "timestamp": list(ts) * 2})
    events = pd.DataFrame([
        {"station_id": "x", "source": "a", "start": ts[0], "end": ts[2], "fault": "reviewed_event"},
        {"station_id": "x", "source": "b", "start": ts[0], "end": ts[2], "fault": "reviewed_event"},
        {"station_id": "missing", "source": "a", "start": ts[0], "end": ts[2], "fault": "reviewed_event"}])
    result = per_fault_breakdown(events, frame, np.array([0, 0, 1, 0, 0, 0, 0, 1]))
    assert result["reviewed_event"] == {"n_events": 2, "detected": 1, "recall": .5}


def test_unknown_labels_do_not_become_negative_examples():
    s = np.array([0., 1., 1000., -1000.])
    y = np.array([0., 1., np.nan, -1.])
    threshold, detail = choose_threshold(s, y)
    assert threshold == 1 and detail["pw"]["f1"] == 1
    assert detail["n_labeled"] == 2
    report = _binary_report(y, s, np.array([0, 1, 1, 0]))
    assert report["n_unknown_labels"] == 2
    assert report["pointwise"]["f1"] == report["pr_auc"] == 1


def test_threshold_search_is_not_restricted_to_assumed_anomaly_rate():
    # The former 90th-percentile floor cannot detect this entire positive set.
    y = np.r_[np.zeros(100), np.ones(100)]
    scores = np.arange(200.)
    threshold, detail = choose_threshold(scores, y)
    assert threshold == 100 and detail["pw"]["f1"] == 1


def test_group_separators_do_not_change_prevalence_or_auc():
    y = np.array([0, 1, 1, 0])
    s = np.array([.2, .8, .1, .4])
    pred = (s > .5).astype(int)
    plain = _binary_report(y, s, pred)
    grouped = _binary_report(y, s, pred, group_ids=np.array(["a", "a", "b", "b"]))
    for key in ("n_rows", "anomaly_rate_true", "anomaly_rate_pred", "pr_auc", "roc_auc"):
        assert grouped[key] == plain[key]
    assert grouped["point_adjust"]["recall"] == .5
    assert len(grouped["point_f1_group_bootstrap_95ci"]) == 2


def test_unknown_gap_breaks_events_without_becoming_truth():
    y = np.array([1., np.nan, 1.])
    report = _binary_report(y, np.array([1., 0., 0.]), np.array([1, 0, 0]))
    assert report["point_adjust"]["recall"] == .5
    assert report["anomaly_rate_true"] == 1


def test_purged_partitions_are_chronological_and_exclude_shared_events():
    ts = pd.date_range("2020-01-01", periods=2000, freq="h", tz="UTC")
    y = np.zeros(2000)
    y[1000:1300] = 1  # spans fit/tune, including the purge gap
    y[1500:1700] = 1  # spans tune/calibrate
    y[::19] = np.nan
    # Use a complete known event crossing fit/tune to exercise event purging.
    y[1000:1300] = 1
    parts = temporal_partitions(ts, y)
    fit, tune, cal = (parts[k] for k in ("fit", "tune", "calibrate"))
    assert ts[tune[0]] - ts[fit[-1]] > pd.Timedelta(hours=168)
    assert ts[cal[0]] - ts[tune[-1]] > pd.Timedelta(hours=168)
    assert not np.intersect1d(fit, tune).size
    assert not np.intersect1d(tune, cal).size
    for rows in parts.values():
        assert np.isin(y[rows], [0, 1]).all()
        assert not np.any((rows >= 1000) & (rows < 1300))


def test_purge_uses_elapsed_time_and_rejects_reversed_timestamps():
    ts = pd.date_range("2020-01-01", periods=2000, freq="30min", tz="UTC")
    parts = temporal_partitions(ts, np.zeros(2000))
    assert ts[parts["tune"][0]] - ts[parts["fit"][-1]] > pd.Timedelta(hours=168)
    with pytest.raises(ValueError, match="chronological"):
        temporal_partitions(ts[::-1], np.zeros(2000))


def test_station_class_weights_do_not_let_large_stations_dominate():
    groups = np.array(["a"] * 4 + ["b"] * 100)
    y = np.array([0, 0, 0, 1] + [0] * 90 + [1] * 10)
    weights = balanced_group_class_weights(y, groups)
    assert weights[groups == "a"].sum() == pytest.approx(weights[groups == "b"].sum())
    assert weights[(groups == "a") & (y == 1)].sum() == pytest.approx(
        weights[(groups == "a") & (y == 0)].sum())


def test_local_loader_refuses_legacy_before_opening_parquet(tmp_path, monkeypatch):
    from awsad.train_pipeline import load_processed
    from awsad.data.training_contract import TrainingDataContractError
    (tmp_path / "source_metadata.json").write_text('{"status":"legacy_local_snapshot"}')
    def forbidden_read(*args, **kwargs):
        pytest.fail("must refuse before reading training tables")
    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    with pytest.raises(TrainingDataContractError):
        load_processed(tmp_path)
