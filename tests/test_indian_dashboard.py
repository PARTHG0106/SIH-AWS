"""Software fixtures only; they are not demonstration source observations."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from app import indian_dashboard as ui
from awsad.demo.indian_stations import simulate_scenario


@pytest.fixture
def source_fixture():
    times = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z",
                            "2024-01-01T03:00:00Z", "2024-01-01T06:00:00Z",
                            "2024-01-01T09:00:00Z", "2024-01-03T00:00:00Z"], utc=True)
    frame = pd.DataFrame({
        "timestamp": times, "station_id": "test", "source_row": [2, 3, 4, 5, 6, 7],
        "baseline_temperature_c": [20., 21., 21.5, np.nan, 23., 24.],
        "derived_relative_humidity_pct": [60., 62., 63., 65., 67., 66.],
        "baseline_sea_level_pressure_hpa": [1010., 1011., 1011.5, 1012., 1011., 1010.],
        "raw_file_sha256": "a" * 64, "raw_TMP": ["+0200,1"] * 6,
        "source": "software_test_fixture", "hardware_fault_status": "unknown",
        "eligible_for_real_training": False,
    })
    return frame


@pytest.fixture
def ui_bundle(tmp_path, source_fixture, monkeypatch):
    frame = source_fixture
    catalog = pd.DataFrame([{
        "station_id": "test", "name": "SOFTWARE TEST FIXTURE", "latitude": 28.5,
        "longitude": 77.2, "elevation_m": 200., "year": 2024, "rows": len(frame),
        "start_utc": frame.timestamp.min(), "end_utc": frame.timestamp.max(),
    }])
    bundle = {"root": tmp_path, "catalog": catalog, "manifest": {
        "artifact_kind": "indian_station_demo", "dataset_policy": "demo_only_not_real_training",
        "eligible_for_real_training": False, "software_test_fixture": True,
    }}
    monkeypatch.setattr(ui, "cached_indian_bundle", lambda *args: bundle)
    monkeypatch.setattr(ui, "cached_indian_station", lambda *args: frame.copy())
    monkeypatch.setenv("SKYGUARD_INDIAN_DEMO", str(tmp_path))
    return bundle


def _test_app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string(
        "from pathlib import Path\n"
        "from app.indian_dashboard import render_indian_dashboard\n"
        f"render_indian_dashboard(Path({str(ROOT)!r}))\n",
        default_timeout=30,
    ).run()


def test_scenario_counts_differences_and_missingness_without_losing_duplicates(source_fixture):
    before = source_fixture.copy(deep=True)
    frame = simulate_scenario(source_fixture, channel="baseline_temperature_c", scenario="dropout",
                              start=pd.Timestamp("2024-01-01T03:00:00Z"), duration_hours=4., magnitude=0.)
    assert ui.scenario_summary(frame) == {
        "records": 6, "modified_records": 2, "modified_values": 2,
        "missing_baseline_values": 1, "duplicate_timestamp_rows": 2,
    }
    pd.testing.assert_frame_equal(source_fixture, before)
    assert frame.hardware_fault_status.eq("unknown").all()
    assert not frame.eligible_for_real_training.any()


def test_scenario_chart_never_fills_values_or_collapses_duplicate_reports(source_fixture):
    frame = simulate_scenario(source_fixture, channel="baseline_temperature_c", scenario="dropout",
                              start=pd.Timestamp("2024-01-01T03:00:00Z"), duration_hours=4., magnitude=0.)
    chart = ui.scenario_chart_data(frame, "baseline_temperature_c")
    baseline = chart.loc[chart.series.eq("Source temperature")]
    synthetic = chart.loc[chart.series.eq(ui.SYNTHETIC_LABEL)]
    markers = chart.loc[chart.series.eq(ui.APPLIED_LABEL)]
    assert baseline.value.tolist() == [20., 21., 21.5, 23., 24.]
    assert baseline.source_row.tolist() == [2, 3, 4, 6, 7]
    assert synthetic.value.tolist() == [20., 23., 24.]
    assert markers.source_row.tolist() == [3, 4]
    assert set(chart.timestamp) <= set(source_fixture.timestamp)
    assert "segment" not in chart  # Presentation uses points, never gap-bridging lines.
    humidity = ui.scenario_chart_data(frame, "derived_relative_humidity_pct")
    assert ui.APPLIED_LABEL not in humidity.series.values
    assert "RH derived from source T / dew point" in humidity.series.values


def test_indian_page_renders_scenarios_and_an_empty_window(ui_bundle):
    app = _test_app()
    assert not app.exception
    assert app.title[0].value == "India · Station scenarios"
    assert any("SYNTHETIC DEMONSTRATION" in item.value for item in app.warning)
    assert [metric.value for metric in app.metric] == ["6", "2", "2", "1"]
    assert len(app.get("vega_lite_chart")) == 3
    assert len(app.get("download_button")) == 1
    assert any("calculated from source temperature" in item.value for item in app.markdown)
    assert any(expander.label == "Source records & provenance" for expander in app.expander)
    app.selectbox(key="indian_scenario").set_value("baseline").run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["6", "0", "0", "1"]
    app.selectbox(key="indian_window_days").set_value(1).run()
    app.date_input[0].set_value(pd.Timestamp("2024-01-02").date()).run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["0", "0", "0", "0"]
    assert not app.get("vega_lite_chart")
    assert not app.get("download_button")
    assert any("No source records" in item.value for item in app.info)


def test_all_missing_source_channel_stays_unavailable(ui_bundle, source_fixture, monkeypatch):
    source_fixture["baseline_sea_level_pressure_hpa"] = np.nan
    monkeypatch.setattr(ui, "cached_indian_station", lambda *args: source_fixture.copy())
    app = _test_app()
    app.selectbox(key="indian_channel").set_value("baseline_sea_level_pressure_hpa").run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["6", "0", "0", "7"]
    assert len(app.get("vega_lite_chart")) == 2
    assert any("No baseline or synthetic values" in item.value for item in app.info)


def test_missing_indian_artifact_reports_a_builder_command(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYGUARD_INDIAN_DEMO", str(tmp_path / "missing"))
    app = _test_app()
    assert not app.exception
    assert len(app.error) == 1
    assert "Cannot load the Indian demonstration" in app.error[0].value
    assert any("scripts/build_indian_demo.py" in item.value for item in app.code)
    assert not app.get("vega_lite_chart")


def test_completed_indian_bundle_in_main_dashboard(monkeypatch, tmp_path):
    """Opt-in read-only integration against the actual separately built artifact."""
    actual = os.environ.get("SKYGUARD_TEST_INDIAN_DEMO")
    if not actual:
        pytest.skip("Set SKYGUARD_TEST_INDIAN_DEMO to verify a completed Indian demo bundle")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SKYGUARD_INDIAN_DEMO", str(Path(actual).resolve()))
    # India must work even without any USA artifacts.
    monkeypatch.setenv("SKYGUARD_ARTIFACTS", str(tmp_path / "no_usa_artifacts"))
    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    app.selectbox(key="dashboard_page").set_value("India · Synthetic scenarios").run()
    assert not app.exception
    assert app.title[0].value == "India · Station scenarios"
    assert not app.error
    bundle = ui.load_demo_bundle(actual)
    for station_id in bundle["catalog"].station_id.astype(str):
        app.selectbox(key="indian_station").set_value(station_id).run()
        assert not app.exception
        assert not app.error
        assert len(app.get("vega_lite_chart")) == 3
    for scenario in ui.SCENARIO_LABELS:
        app.selectbox(key="indian_scenario").set_value(scenario).run()
        assert not app.exception
        assert not app.error
    app.selectbox(key="dashboard_page").set_value("USA · Real observations").run()
    assert not app.exception
    assert app.title[0].value == "Observation replay"
    assert len(app.error) == 1  # Missing USA artifacts remain an explicit error.
