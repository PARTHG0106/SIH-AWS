"""Software fixtures only: strict lineage checks across pandas time units."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from awsad.data.verify_surfrad import _assert_observation_frames_equal


def _frame(unit):
    return pd.DataFrame({
        "timestamp": pd.Series(pd.to_datetime(["2024-01-01T00:00:00Z"]), dtype=f"datetime64[{unit}, UTC]"),
        "temperature_c": [4.5],
        "label": pd.array([pd.NA], dtype="Int8"),
    })


@pytest.mark.parametrize("actual_unit,expected_unit", [("us", "ns"), ("ns", "us")])
def test_same_utc_instants_at_different_resolutions_pass_without_mutation(actual_unit, expected_unit):
    actual, expected = _frame(actual_unit), _frame(expected_unit)
    before_actual, before_expected = actual.copy(), expected.copy()
    _assert_observation_frames_equal(actual, expected)
    pd.testing.assert_frame_equal(actual, before_actual, check_exact=True)
    pd.testing.assert_frame_equal(expected, before_expected, check_exact=True)


def test_one_nanosecond_timestamp_shift_is_rejected():
    actual, expected = _frame("ns"), _frame("us")
    actual.loc[0, "timestamp"] += pd.Timedelta(1, unit="ns")
    with pytest.raises(AssertionError):
        _assert_observation_frames_equal(actual, expected)


@pytest.mark.parametrize("change", ["naive", "non_utc", "string"])
def test_timestamp_must_remain_a_utc_aware_datetime(change):
    actual, expected = _frame("us"), _frame("ns")
    if change == "naive":
        actual["timestamp"] = actual["timestamp"].dt.tz_localize(None)
    elif change == "non_utc":
        actual["timestamp"] = actual["timestamp"].dt.tz_convert("Asia/Kolkata")
    else:
        actual["timestamp"] = actual["timestamp"].astype(str)
    with pytest.raises(ValueError, match="UTC-aware"):
        _assert_observation_frames_equal(actual, expected)


@pytest.mark.parametrize("change", ["temperature", "label", "temperature_dtype", "label_dtype"])
def test_other_values_and_dtypes_still_match_exactly(change):
    actual, expected = _frame("us"), _frame("ns")
    if change == "temperature":
        actual.loc[0, "temperature_c"] = 4.5000000001
    elif change == "label":
        actual.loc[0, "label"] = 0
    elif change == "temperature_dtype":
        actual["temperature_c"] = actual["temperature_c"].astype("float32")
    else:
        actual["label"] = actual["label"].astype("Int64")
    with pytest.raises(AssertionError):
        _assert_observation_frames_equal(actual, expected)
