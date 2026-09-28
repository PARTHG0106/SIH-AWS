"""Synthetic software fixtures only; none are dataset observations or labels."""
import json

import pandas as pd
import pytest

from awsad.evaluation.real_events import (
    evaluate_reviewed_events, extract_candidate_events, review_template,
    sample_background_candidates, validate_reviews,
)


def decisions(minutes=(0, 1, 2, 3, 4, 6), alerts=(False, True, True, False, None, True)):
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    return pd.DataFrame({
        "group": "test|fixture", "channel": "temperature_c", "split": "test",
        "timestamp": [start + pd.Timedelta(minutes=m) for m in minutes],
        "observation_id": [f"test-only-{m}" for m in minutes], "is_alert": list(alerts),
        "score": [float(m) for m in minutes], "reason_codes": ['["test_only"]'] * len(minutes),
    })


def reviewed(status, start, end, **changes):
    value = {
        "review_id": "review-" + str(start), "event_id": "event-" + str(start),
        "group": "test|fixture", "channel": "temperature_c",
        "start_utc": f"2025-01-01T00:{start:02d}:00Z", "end_utc": f"2025-01-01T00:{end:02d}:00Z",
        "review_status": status, "onset_earliest_utc": f"2025-01-01T00:{start:02d}:00Z",
        "onset_latest_utc": f"2025-01-01T00:{start:02d}:00Z", "reviewer": "TEST FIXTURE ONLY",
        "reviewed_at_utc": "2025-01-02T00:00:00Z", "evidence_basis": "operator_log",
        "evidence_refs": json.dumps([{"url": "https://example.invalid/test-fixture", "sha256": "a" * 64}]),
        "review_notes": "Software test fixture; never use as ground truth.",
    }
    value.update(changes)
    return value


def test_unreviewed_flags_are_never_real_fault_or_normal_truth():
    report = evaluate_reviewed_events(decisions(), review_template())
    assert report["status"] == "unverified_no_reviewed_truth"
    assert report["event_recall"] is None
    assert report["false_alert_fraction"] is None
    assert report["false_alert_episodes"] is None
    assert report["unknown_alert_rows"] == 3
    assert report["unknown_observation_rows"] == 6
    json.dumps(report, allow_nan=False)


def test_candidates_break_at_gaps_false_missing_splits_and_groups():
    frame = decisions()
    events = extract_candidate_events(frame)
    assert events["observation_count"].tolist() == [2, 1]
    assert events["first_observation_id"].tolist() == ["test-only-1", "test-only-6"]
    assert events["review_status"].eq("unknown").all()
    frame.loc[2, "split"] = "holdout"
    assert len(extract_candidate_events(frame)) == 3
    second = frame.copy()
    second["group"] = "other|fixture"
    second["observation_id"] += "-other"
    assert len(extract_candidate_events(pd.concat([frame, second]))) == 6


def test_review_template_does_not_inherit_a_candidate_label_or_onset():
    events = extract_candidate_events(decisions())
    events["review_status"] = "confirmed_event"
    template = review_template(events)
    assert template["review_status"].eq("unknown").all()
    assert template["onset_earliest_utc"].isna().all()
    assert template.iloc[0]["end_utc"] == "2025-01-01T00:03:00+00:00"
    assert evaluate_reviewed_events(decisions(), template)["event_recall"] is None


def test_event_recall_and_bounded_delay_do_not_use_unknown_alerts():
    reviews = pd.DataFrame([
        reviewed("confirmed_event", 1, 3, onset_earliest_utc="2025-01-01T00:00:00Z"),
        reviewed("confirmed_event", 3, 5),
        reviewed("confirmed_event", 10, 12),
    ])
    report = evaluate_reviewed_events(decisions(), reviews)
    assert report["event_recall"] == pytest.approx(1 / 3)
    assert report["events_without_scored_observations"] == 1
    assert report["mean_detection_delay_seconds_min"] == 0
    assert report["mean_detection_delay_seconds_max"] == 60
    assert report["unknown_alert_rows"] == 1
    assert report["false_alert_fraction"] is None
    assert report["events"][0]["first_alert_observation_id"] == "test-only-1"


def test_false_alarms_use_only_reviewed_scored_background_and_do_not_fill_gaps():
    frame = decisions()
    reviews = pd.DataFrame([reviewed("reviewed_background", 0, 5)])
    report = evaluate_reviewed_events(frame, reviews)
    assert report["background_scored_rows"] == 4  # the missing decision is not negative
    assert report["false_alert_rows"] == 2
    assert report["false_alert_fraction"] == 0.5
    assert report["false_alert_episodes"] == 1
    assert report["background_contiguous_scored_seconds"] == 180
    assert report["unknown_alert_rows"] == 1
    one = evaluate_reviewed_events(frame, pd.DataFrame([reviewed("reviewed_background", 6, 7)]))
    assert one["false_alert_rows"] == 1
    assert one["false_alert_episodes_per_24_contiguous_scored_hours"] is None


def test_background_sampling_proposes_actual_rows_and_never_normal():
    frame = decisions(range(20), [False] * 20)
    proposals = sample_background_candidates(frame, max_per_group=2, window_minutes=3)
    assert len(proposals) == 2
    assert proposals["review_status"].eq("unknown").all()
    assert proposals["candidate_kind"].eq("non_alert_review_sample").all()
    assert proposals["first_observation_id"].tolist() == ["test-only-0", "test-only-18"]
    assert evaluate_reviewed_events(frame, review_template(proposals))["false_alert_rows"] is None


@pytest.mark.parametrize("change", [
    {"evidence_basis": "provider_qc"}, {"evidence_basis": "model_prediction"},
    {"evidence_refs": "[]"}, {"evidence_refs": '[{"url":"https://example.invalid", "sha256":"bad"}]'},
    {"reviewer": None}, {"reviewed_at_utc": "2025-01-02"},
    {"onset_earliest_utc": None}, {"onset_latest_utc": "2025-01-01T00:02:00Z"},
])
def test_acceptance_requires_explicit_independent_review(change):
    with pytest.raises(ValueError):
        validate_reviews(pd.DataFrame([reviewed("confirmed_event", 1, 3, **change)]))


def test_contradictory_intervals_and_duplicate_event_ids_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        validate_reviews(pd.DataFrame([reviewed("confirmed_event", 0, 3), reviewed("reviewed_background", 2, 4)]))
    with pytest.raises(ValueError, match="counted twice"):
        validate_reviews(pd.DataFrame([reviewed("confirmed_event", 0, 3), reviewed("confirmed_event", 4, 5, event_id="event-0")]))


def test_duplicate_and_string_decisions_are_rejected():
    frame = decisions()
    with pytest.raises(ValueError, match="Duplicate"):
        extract_candidate_events(pd.concat([frame, frame.iloc[:1]]))
    frame["is_alert"] = "False"
    with pytest.raises(ValueError, match="booleans"):
        evaluate_reviewed_events(frame, review_template())


def test_naive_times_are_rejected_and_inputs_not_mutated():
    frame = decisions()
    original = frame.copy(deep=True)
    extract_candidate_events(frame)
    pd.testing.assert_frame_equal(frame, original)
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="offset"):
        extract_candidate_events(frame)


def test_half_open_intervals_and_group_channel_isolation():
    reviews = pd.DataFrame([reviewed("confirmed_event", 0, 1)])
    assert evaluate_reviewed_events(decisions(), reviews)["event_recall"] == 0
    reviews.loc[0, "channel"] = "pressure_hpa"
    report = evaluate_reviewed_events(decisions(), reviews)
    assert report["events_without_scored_observations"] == 1
    assert report["unknown_observation_rows"] == 6
