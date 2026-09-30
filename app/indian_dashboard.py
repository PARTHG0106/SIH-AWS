"""An explicitly synthetic scenario page on archived Indian station reports.

This page never trains, scores, repairs, or overwrites source observations.
Points preserve native timestamps and duplicate reports without implying cadence.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from awsad.demo.indian_stations import load_demo_bundle, read_demo_station, simulate_scenario
from app.theme import palette, inject_css, configure_chart


CHANNELS = {
    "baseline_temperature_c": "synthetic_temperature_c",
    "derived_relative_humidity_pct": "synthetic_relative_humidity_pct",
    "baseline_sea_level_pressure_hpa": "synthetic_sea_level_pressure_hpa",
}
CHANNEL_LABELS = {
    "baseline_temperature_c": "Temperature (°C)",
    "derived_relative_humidity_pct": "Relative humidity · derived from T / dew point (%)",
    "baseline_sea_level_pressure_hpa": "Sea-level pressure (hPa)",
}
SOURCE_LABELS = {
    "baseline_temperature_c": "Source temperature",
    "derived_relative_humidity_pct": "RH derived from source T / dew point",
    "baseline_sea_level_pressure_hpa": "Source sea-level pressure",
}
SCENARIO_LABELS = {
    "baseline": "Baseline · unchanged copy",
    "spike": "Spike · one report time",
    "drift": "Drift · increasing offset",
    "stuck": "Stuck · hold one value",
    "dropout": "Dropout · remove values",
}
SYNTHETIC_LABEL = "Synthetic scenario copy"
APPLIED_LABEL = "Applied scenario"


@st.cache_resource(show_spinner=False)
def cached_indian_bundle(directory: str, revision: int) -> dict:
    return load_demo_bundle(directory)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_indian_station(directory: str, revision: int, station_id: str) -> pd.DataFrame:
    return read_demo_station(cached_indian_bundle(directory, revision), station_id)


def scenario_summary(frame: pd.DataFrame) -> dict[str, int]:
    """Count actual differences, including removed values, without fault labels."""
    changed = pd.DataFrame(index=frame.index)
    for baseline, synthetic in CHANNELS.items():
        left, right = frame[baseline], frame[synthetic]
        changed[baseline] = ~(left.eq(right) | (left.isna() & right.isna()))
    return {
        "records": len(frame),
        "modified_records": int(changed.any(axis=1).sum()),
        "modified_values": int(changed.sum().sum()),
        "missing_baseline_values": int(frame[list(CHANNELS)].isna().sum().sum()),
        "duplicate_timestamp_rows": int(frame.timestamp.duplicated(keep=False).sum()),
    }


def scenario_chart_data(frame: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Return finite points only, without inserting, aggregating, or joining rows.

    Applied-scenario markers sit at the baseline value, so removals remain
    visible even when the synthetic value is missing. The marker is a user
    scenario annotation, never a detector result or a fault label.
    """
    rows = []
    for column, label in [(channel, SOURCE_LABELS[channel]),
                          (CHANNELS[channel], SYNTHETIC_LABEL)]:
        values = pd.to_numeric(frame[column], errors="coerce")
        present = values.notna() & np.isfinite(values)
        view = pd.DataFrame({
            "timestamp": frame.timestamp,
            "timestamp_utc": frame.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "value": values,
            "series": label,
            "source_row": frame["source_row"] if "source_row" in frame else frame.index,
        })
        rows.append(view.loc[present])
    if "scenario_applied" in frame:
        applied = frame.scenario_applied.eq(True).fillna(False)
        if "scenario_channel" in frame:
            applied &= frame.scenario_channel.eq(channel)
        markers = rows[0].loc[applied.reindex(rows[0].index).fillna(False)].copy()
        markers["series"] = APPLIED_LABEL
        rows.append(markers)
    return pd.concat(rows, ignore_index=True)


def render_scenario_channel(frame: pd.DataFrame, channel: str, p: dict) -> None:
    st.markdown(f"**{CHANNEL_LABELS[channel]}**")
    values = scenario_chart_data(frame, channel)
    if values.empty:
        st.info("No baseline or synthetic values are available for this channel in the selected window.")
        return
    labels = [SOURCE_LABELS[channel], SYNTHETIC_LABEL, APPLIED_LABEL]
    points = alt.Chart(values).mark_point(strokeWidth=1.8, opacity=0.92).encode(
        x=alt.X("timestamp:T", title="Original report time · UTC", scale=alt.Scale(type="utc")),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=labels, range=[p["ink"], p["accent"], p["alert"]]),
            legend=alt.Legend(title=None, orient="top", labelLimit=430)),
        shape=alt.Shape("series:N", scale=alt.Scale(
            domain=labels, range=["circle", "cross", "diamond"]), legend=None),
        size=alt.Size("series:N", scale=alt.Scale(
            domain=labels, range=[48, 82, 168]), legend=None),
        tooltip=[alt.Tooltip("timestamp_utc:N", title="Time"),
                 alt.Tooltip("source_row:N", title="Original source row"),
                 alt.Tooltip("series:N", title="Value origin"),
                 alt.Tooltip("value:Q", title="Value", format=".3f")],
    ).interactive()
    st.altair_chart(configure_chart(points, p, height=214), width="stretch")


def _default_window_date(frame: pd.DataFrame):
    """Open a recorded day with the most available baseline values."""
    counts = frame[list(CHANNELS)].notna().sum(axis=1)
    by_day = counts.groupby(frame.timestamp.dt.date).sum()
    return by_day.idxmax()


def render_indian_dashboard(root: Path) -> None:
    p = palette()
    inject_css(p)
    st.title("India · Station scenarios")
    st.caption("SkyGuard AI · Archived NOAA ISD reports from Indian stations · All times UTC")
    st.warning("SYNTHETIC DEMONSTRATION · The scenario series contains changes you choose. "
               "It is excluded from real-data training and evaluation. Actual hardware-fault status is unknown.")
    st.write("Compare the archived reports with a separate synthetic copy. Temperature and sea-level "
             "pressure come from the source; relative humidity is calculated from source temperature "
             "and dew point. These are historical station reports, not a live IMD AWS feed.")

    default_directory = os.environ.get("SKYGUARD_INDIAN_DEMO", str(root / "data" / "indian_demo_20260929"))
    with st.sidebar.expander("Indian demo files"):
        directory_text = st.text_input("Indian artifact directory", value=default_directory,
                                       key="indian_artifact_directory")
        if st.button("Reload Indian artifacts", key="indian_reload"):
            st.session_state.indian_artifact_revision = st.session_state.get("indian_artifact_revision", 0) + 1
    revision = st.session_state.get("indian_artifact_revision", 0)
    directory = Path(directory_text).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    try:
        bundle = cached_indian_bundle(str(directory), revision)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Cannot load the Indian demonstration: {exc}")
        st.markdown("Build the separate demonstration artifacts from the archived Indian source files:")
        st.code("python scripts/build_indian_demo.py --out data/indian_demo_20260929", language="shell")
        return

    catalog = bundle["catalog"].copy()
    if catalog.empty:
        st.error("The Indian demonstration station catalog is empty.")
        return
    catalog["station_id"] = catalog.station_id.astype(str)
    station_lookup = catalog.set_index("station_id")
    station_id = st.sidebar.selectbox(
        "Indian source station", catalog.station_id.tolist(), key="indian_station",
        format_func=lambda value: f"{station_lookup.loc[value, 'name']} · {value}",
    )
    station = station_lookup.loc[station_id]
    try:
        source = cached_indian_station(str(directory), revision, station_id)
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Cannot read this station's archived reports: {exc}")
        return
    if source.empty:
        st.warning("This source station contains no archived reports.")
        return
    st.sidebar.caption(f"{len(source):,} source records · {source.timestamp.min():%d %b %Y} — "
                       f"{source.timestamp.max():%d %b %Y}\n\nPublished station name and ID are retained.")
    date = st.sidebar.date_input(
        "Indian window start (UTC)", value=_default_window_date(source),
        min_value=source.timestamp.min().date(), max_value=source.timestamp.max().date(),
        key=f"indian_window_date_{station_id}",
    )
    days = st.sidebar.selectbox("Indian window length (days)", [3, 1, 7], key="indian_window_days")
    start = pd.Timestamp(datetime.combine(date, time.min, tzinfo=timezone.utc))
    end = start + pd.Timedelta(days=days)
    window = source.loc[source.timestamp.ge(start) & source.timestamp.lt(end)].copy()
    scenario = st.sidebar.selectbox("Scenario", list(SCENARIO_LABELS), index=1,
                                    format_func=SCENARIO_LABELS.get, key="indian_scenario")
    channel = st.sidebar.selectbox("Scenario variable", list(CHANNELS),
                                   format_func=CHANNEL_LABELS.get, key="indian_channel",
                                   disabled=scenario == "baseline")
    scenario_start = start
    duration = 12.0
    magnitude = 0.0
    if not window.empty and scenario != "baseline":
        timestamps = window.timestamp.drop_duplicates().tolist()
        scenario_start = st.sidebar.selectbox(
            "Scenario start (UTC)", timestamps, index=len(timestamps) // 3,
            format_func=lambda value: value.strftime("%d %b %H:%M:%S"),
            key=f"indian_scenario_start_{station_id}_{date}_{days}",
        )
        duration = st.sidebar.select_slider("Scenario interval (hours)", [1, 3, 6, 12, 24, 48],
                                            value=12, key="indian_duration")
        if scenario in ("spike", "drift"):
            unit = {"baseline_temperature_c": "°C", "derived_relative_humidity_pct": "percentage points",
                    "baseline_sea_level_pressure_hpa": "hPa"}[channel]
            magnitude = st.sidebar.number_input(
                f"{'Spike offset' if scenario == 'spike' else 'Offset at interval end'} ({unit})",
                min_value=-100.0, max_value=100.0, value=8.0, step=1.0,
                key=f"indian_magnitude_{channel}",
            )
        explanations = {
            "spike": "Adds an offset at the first available report time in the interval. Duplicate reports at that time are retained.",
            "drift": "Increases the offset from zero toward the chosen end offset across the interval. Only existing values change.",
            "stuck": "Holds the first available value in the interval. Source gaps and missing values stay missing.",
            "dropout": "Removes existing values in the interval from the synthetic copy. Source values remain available for comparison.",
        }
        st.sidebar.caption(explanations[scenario])
    st.sidebar.caption("Native report times are preserved. This view does not run a detector or create real fault labels.")

    scenario_ready = True
    if window.empty:
        simulated = window.copy()
        for baseline, synthetic in CHANNELS.items():
            simulated[synthetic] = simulated[baseline].copy()
    else:
        finite = np.isfinite(pd.to_numeric(window[channel], errors="coerce"))
        interval = window.timestamp.ge(scenario_start) & window.timestamp.lt(
            scenario_start + pd.Timedelta(hours=float(duration)))
        scenario_ready = scenario == "baseline" or bool((finite & interval).any())
        if not scenario_ready:
            st.info("The selected channel has no available source value in this scenario interval. "
                    "No scenario is applied; the charts show an unchanged copy. Choose another channel or interval to export a scenario.")
        try:
            simulated = simulate_scenario(window, channel=channel,
                                          scenario=scenario if scenario_ready else "baseline", start=scenario_start,
                                          duration_hours=float(duration), magnitude=float(magnitude))
        except (ValueError, KeyError) as exc:
            st.error(f"Cannot apply this scenario: {exc}")
            return
    summary = scenario_summary(simulated)
    cards = st.columns(4)
    cards[0].metric("Source records", f"{summary['records']:,}",
                    help="Original archived rows in this window; duplicate report times each remain a row.")
    cards[1].metric("Modified records", f"{summary['modified_records']:,}",
                    help="Rows whose synthetic copy differs in at least one value. This is a chosen simulation, not detected faults.")
    cards[2].metric("Modified values", f"{summary['modified_values']:,}",
                    help="Individual values changed or removed in the synthetic copy. Unchanged copies and pre-existing missing values do not count.")
    cards[3].metric("Missing baseline values", f"{summary['missing_baseline_values']:,}",
                    help="Missing temperature, source-derived RH, or sea-level pressure values among existing rows; one row can contribute three.")
    st.caption(f"{station['name']} · {station_id} · {start:%d %b %Y} through "
               f"{(end - pd.Timedelta(days=1)):%d %b %Y} UTC · "
               f"{summary['duplicate_timestamp_rows']:,} rows share a timestamp with another report.")
    with st.expander("How to read this demonstration"):
        st.markdown("**Circles:** source temperature / sea-level pressure, or RH derived from source T and dew point. "
                    "**Terracotta crosses:** the synthetic scenario copy. **Crimson diamonds:** existing values where the chosen "
                    "scenario was applied, drawn at the baseline value so a dropout remains visible. A scenario can be applied "
                    "without changing a value, such as zero drift at its start.")
        st.markdown("Every point uses an original report timestamp. Reports may be irregular or duplicated, so points are "
                    "never joined by lines and no missing-minute count is inferred. Missing source values stay missing. "
                    "The four cards describe this window; none is an accuracy, detection, or confirmed-fault count.")
        st.markdown("**Baseline** makes an unchanged copy; it does not assert that the original reports are fault-free. "
                    "An offset can make the synthetic copy physically implausible. That is a scenario parameter, not an observed reading.")

    if simulated.empty:
        st.info("No source records exist in this calendar window. Choose another date or a longer window.")
    else:
        if scenario_ready and scenario != "baseline" and not summary["modified_records"]:
            st.info("The current scenario changes no values in this window. The selected channel may be missing, "
                    "the interval may contain no usable value, or the chosen offset/held value may leave it unchanged.")
        for selected_channel in CHANNELS:
            render_scenario_channel(simulated, selected_channel, p)
        if scenario_ready:
            download = simulated.copy()
            download["demo_notice"] = "SYNTHETIC DEMONSTRATION; not real observations or fault ground truth"
            st.download_button("Download labelled scenario CSV", download.to_csv(index=False).encode("utf-8"),
                               file_name=f"SYNTHETIC_india_{station_id}_{date}_{scenario}.csv", mime="text/csv",
                               key="indian_download")

    with st.expander("Source records & provenance"):
        st.write(f"Artifact directory: `{bundle['root']}`")
        st.markdown("Original source fields, quality codes, source-row identifiers and hashes remain in the download. "
                    "Provider QC is evidence about reported data quality; it does not confirm hardware causes. "
                    "This page makes no claim of independent RH measurement or station-pressure measurement, "
                    "nor of verified IMD AWS sensor hardware.")
        visible_columns = [name for name in ["timestamp", "source_row", *CHANNELS, *CHANNELS.values(),
                           "scenario_type", "scenario_channel", "scenario_applied", "injected_demo_change",
                           "hardware_fault_status", "eligible_for_real_training"] if name in simulated]
        st.dataframe(simulated[visible_columns], hide_index=True, width="stretch")
        st.caption("Artifact manifest · includes original-file and station-file SHA-256 hashes")
        st.json(bundle["manifest"], expanded=False)
    with st.expander("Published station locations"):
        st.caption("Locations are those published for the archived source stations, not invented station locations.")
        coordinates = catalog[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").dropna()
        if not coordinates.empty:
            st.map(coordinates, latitude="latitude", longitude="longitude", color=p["accent2"], size=25000,
                   height=300)
        st.dataframe(catalog[[name for name in ["name", "station_id", "latitude", "longitude", "elevation_m"]
                                 if name in catalog]], hide_index=True, width="stretch")
