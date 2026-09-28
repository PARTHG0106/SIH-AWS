"""SkyGuard real-observation replay. Run: streamlit run app/streamlit_app.py."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import os
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.real_dashboard import (CHANNELS, CHANNEL_LABELS, boolean_flags, chart_series,
                                load_bundle, missing_intervals, read_observations,
                                review_export, window_summary)

st.set_page_config(page_title="SkyGuard · Observation replay", page_icon="🌦", layout="wide")
alt.data_transformers.disable_max_rows()
st.markdown("""
<style>
    .block-container {max-width: 1440px; padding-top: 2rem;}
    [data-testid="stMetric"] {background: #f2f7f6; border: 1px solid #dbe8e4;
        border-radius: 10px; padding: 14px 18px; color: #193e35;}
    [data-testid="stMetricLabel"] {color: #46665e;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def cached_bundle(directory: str, revision: int):
    return load_bundle(directory)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_window(directory: str, revision: int, group: str, start: str, end: str):
    return read_observations(cached_bundle(directory, revision), group, start, end)


def render_channel(frame: pd.DataFrame, channel: str, long_horizon: bool):
    st.markdown(f"**{CHANNEL_LABELS[channel]}**")
    lines = chart_series(frame, channel, include_long_horizon=long_horizon)
    if lines.empty:
        st.info("No measured values or model predictions are available in this window.")
        return
    colors = alt.Scale(domain=["Observed", "Model: 1-minute horizon", "Model: 60-minute horizon"],
                       range=["#116e63", "#dc8b29", "#6978bb"])
    base = alt.Chart(lines).mark_line(strokeWidth=1.7, point={"size": 6}).encode(
        x=alt.X("timestamp:T", title="UTC", scale=alt.Scale(type="utc")),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=colors, legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.StrokeDash("series:N", legend=None), detail="segment:N",
        tooltip=[alt.Tooltip("timestamp:T", title="UTC", format="%Y-%m-%d %H:%M:%S"),
                 alt.Tooltip("series:N", title="Value origin"), alt.Tooltip("value:Q", format=".3f")],
    )
    layers = [base]
    alert_column = f"{channel}__alert"
    if alert_column in frame:
        marker_columns = [name for name in ("timestamp", channel, f"{channel}__reason_codes")
                          if name in frame]
        flagged = frame.loc[boolean_flags(frame[alert_column]) & frame[channel].notna(), marker_columns].copy()
        if not flagged.empty:
            layers.append(alt.Chart(flagged).mark_point(color="#bc4b33", size=48,
                                                        filled=False, strokeWidth=1.5).encode(
                x=alt.X("timestamp:T", scale=alt.Scale(type="utc")), y=alt.Y(f"{channel}:Q"),
                tooltip=[alt.Tooltip("timestamp:T", title="UTC"),
                         alt.Tooltip(f"{channel}:Q", title="Observed"),
                         alt.Tooltip(f"{channel}__reason_codes:N", title="Candidate reason")]
                if f"{channel}__reason_codes" in flagged else ["timestamp:T"],
            ))
    st.altair_chart(alt.layer(*layers).properties(height=190).interactive(), use_container_width=True)


st.sidebar.title("SkyGuard AI")
st.sidebar.caption("SIH26073 · Real-observation detector")
default_directory = os.environ.get("SKYGUARD_ARTIFACTS", str(ROOT / "artifacts_minute_20260928"))
directory_text = st.sidebar.text_input("Artifacts directory", value=default_directory)
directory = Path(directory_text).expanduser()
if not directory.is_absolute():
    directory = ROOT / directory
if "artifact_revision" not in st.session_state:
    st.session_state.artifact_revision = 0
if st.sidebar.button("Reload artifacts"):
    st.session_state.artifact_revision += 1
revision = st.session_state.artifact_revision

st.title("Observation replay")
st.caption("SkyGuard AI · NOAA SURFRAD · Historical native one-minute averages · All times UTC")
st.info("Candidate alerts identify unusual observed behaviour. Hardware-fault status remains unknown "
        "without independent review; provider QC is separate quality evidence.")
try:
    bundle = cached_bundle(str(directory), revision)
except (ValueError, OSError, KeyError) as exc:
    st.error(str(exc))
    st.markdown("Choose a completed minute detector run containing detector.json, metrics.json, "
                "scored observations and candidate_events.csv. No generated feed is substituted "
                "when artifacts are absent.")
    st.stop()

catalog = bundle["catalog"]
group = st.sidebar.selectbox("Station | observation source", catalog["group"].tolist())
station = catalog.loc[catalog["group"].eq(group)].iloc[0]
st.sidebar.caption(f"{int(station['records']):,} archived rows\n\n"
                   f"{station['first']:%d %b %Y} — {station['last']:%d %b %Y}")
events = bundle["events"]
group_events = events.loc[events["group"].eq(group)].copy() if "group" in events else pd.DataFrame()
if "max_score" in group_events:
    group_events = group_events.sort_values("max_score", ascending=False, kind="stable")
modes = ["Calendar window", "Candidate event"] if group_events.empty else ["Candidate event", "Calendar window"]
mode = st.sidebar.radio("Replay selection", modes)
selected_event = None
if mode == "Candidate event" and not group_events.empty:
    event_ids = group_events["event_id"].tolist()
    event_lookup = group_events.set_index("event_id")
    event_id = st.sidebar.selectbox(
        "Candidate to inspect", event_ids,
        format_func=lambda key: f"{event_lookup.loc[key, 'start']:%Y-%m-%d %H:%M} · "
                               f"{event_lookup.loc[key, 'channel']}",
    )
    selected_event = event_lookup.loc[event_id]
    context_minutes = st.sidebar.select_slider("Context before and after (minutes)",
                                               options=[15, 30, 60, 180, 360], value=60)
    start = selected_event["start"] - pd.Timedelta(minutes=context_minutes)
    end = min(selected_event["end"] + pd.Timedelta(minutes=context_minutes),
              start + pd.Timedelta(days=7))
    if selected_event["end"] > end:
        st.sidebar.caption("This long event is shown as a seven-day window. Use the calendar to inspect later dates.")
else:
    if mode == "Candidate event":
        st.sidebar.caption("No candidate events exist for this stream. Showing observed records.")
    date = st.sidebar.date_input("Window start (UTC)", value=station["last"].date(),
                                 min_value=station["first"].date(), max_value=station["last"].date())
    days = st.sidebar.selectbox("Window length (days)", [1, 3, 7])
    start = pd.Timestamp(datetime.combine(date, time.min, tzinfo=timezone.utc))
    end = start + timedelta(days=days) - timedelta(microseconds=1)
long_horizon = st.sidebar.checkbox("Show 60-minute model predictions", value=False)
st.sidebar.caption("Plots preserve absent readings and timestamp gaps. Predictions never replace observations.")

with st.spinner("Reading the selected observed interval…"):
    frame = cached_window(str(directory), revision, group, start.isoformat(), end.isoformat())
summary = window_summary(frame)
columns = st.columns(4)
columns[0].metric("Observed rows in window", f"{summary['records']:,}")
columns[1].metric("Candidate rows", f"{summary['candidates']:,}",
                  help="A detector threshold was crossed; this is not a confirmed hardware fault.")
columns[2].metric("Missing channel readings", f"{summary['missing_values']:,}",
                  help="Absent values among the three channels in existing archived rows.")
columns[3].metric("Unreported minute slots", f"{summary['unreported_slots']:,}",
                  help="Internal timestamp gaps in this window. Archive gaps do not diagnose telemetry failure.")
st.caption(f"{start:%d %b %Y %H:%M} — {end:%d %b %Y %H:%M} UTC · "
           f"{summary['scored']:,} rows have an anomaly score. Missing scores mean no score is available.")

replay_tab, review_tab, evidence_tab = st.tabs(["Readings & reasons", "Candidate review", "Run evidence"])
with replay_tab:
    if frame.empty:
        st.warning("No archived records exist in the selected interval. Choose another interval.")
    else:
        if selected_event is not None:
            st.markdown(f"**Selected candidate:** {selected_event['start']:%Y-%m-%d %H:%M} — "
                        f"{selected_event['end']:%Y-%m-%d %H:%M} UTC")
            st.caption(f"{selected_event.get('channel', '')} · {selected_event.get('reason_codes', '')} · "
                       "Hardware cause: unknown")
        st.caption("Green lines show native measurements; model outputs have separate labelled lines. "
                   "Open red circles mark candidate readings. Gaps break the lines.")
        for channel in CHANNELS:
            render_channel(frame, channel, long_horizon)
        flagged = frame.loc[boolean_flags(frame["is_candidate"])]
        display_columns = [column for column in ["timestamp", "reason_codes", "anomaly_score",
                            "scoring_status", "split", *CHANNELS] if column in frame]
        st.markdown("**Candidate reasons in this window**")
        if flagged.empty:
            st.caption("No candidate readings in this window. This does not establish fault-free operation.")
        else:
            st.dataframe(flagged[display_columns], hide_index=True, use_container_width=True)
        st.caption("Anomaly scores express detector evidence relative to calibrated thresholds; "
                   "they are not probabilities of hardware failure.")
        with st.expander("Provider QC and missing-data details"):
            st.write("SURFRAD code 0 means provider checks passed; codes above 0 flag a quality concern. "
                     "Neither result independently establishes a hardware cause. Missing QC stays unknown.")
            qc_columns = [column for channel in CHANNELS for column in
                          (f"{channel}__raw_qc", f"{channel}__qc_code") if column in frame]
            if qc_columns:
                st.dataframe(frame[["timestamp", *qc_columns]], hide_index=True, use_container_width=True)
            else:
                st.caption("Provider QC fields are unavailable in this export.")
            st.dataframe(missing_intervals(frame), hide_index=True, use_container_width=True)
        with st.expander("Original record and field provenance"):
            index = st.selectbox("Record to inspect", frame.index.tolist(),
                                 format_func=lambda row: f"{frame.loc[row, 'timestamp']:%Y-%m-%d %H:%M:%S} UTC")
            row = frame.loc[index]
            observed_columns = [column for column in frame if column in
                                {"timestamp", "station_id", "source", "observation_id", "raw_file_sha256",
                                 "raw_row_number", "source_url", "retrieved_at_utc", "raw_timestamp",
                                 "native_averaging_seconds", "label", *CHANNELS}
                                or column.startswith("raw_")
                                or any(column.startswith(channel + "__") and not any(
                                    term in column for term in ("prediction", "alert", "score", "reason"))
                                    for channel in CHANNELS)]
            st.dataframe(pd.DataFrame({"field": observed_columns,
                                       "archived value": ["missing / unknown" if pd.isna(row[c]) else str(row[c])
                                                          for c in observed_columns]}),
                         hide_index=True, use_container_width=True)
            st.download_button("Download selected observed window and separate model fields",
                               frame.to_csv(index=False).encode("utf-8"),
                               file_name=f"skyguard_{group.split('|')[0]}_{start:%Y%m%d}_replay.csv",
                               mime="text/csv")

with review_tab:
    st.markdown("**Candidates awaiting independent evidence**")
    st.write("The detector proposes intervals for inspection. A reviewer needs station maintenance, "
             "calibration records or an independently measured comparison before establishing a sensor "
             "fault or cause. A quiet interval is also unreviewed unless evidence establishes otherwise.")
    if group_events.empty:
        st.info("There are no candidate events in this export for the selected station stream.")
    else:
        st.dataframe(group_events, hide_index=True, use_container_width=True)
    reviews = review_export(bundle, group)
    st.download_button("Download event review worksheet", reviews.to_csv(index=False).encode("utf-8"),
                       file_name=f"skyguard_{group.split('|')[0]}_event_review.csv", mime="text/csv")
    st.caption("The worksheet preserves unknown status and evidence references. Downloading or inspecting "
               "an event does not label it. Reviewed evidence must be validated separately before evaluation.")

with evidence_tab:
    st.markdown("**What this run can establish**")
    st.write("This is historical replay of archived observations. Forecast errors, candidate counts and "
             "provider-QC agreement describe different outcomes. Real hardware-fault precision, recall, "
             "F1, detection delay and repair accuracy require an independent labelled benchmark.")
    st.dataframe(catalog.rename(columns={"first": "first_observation_utc", "last": "last_observation_utc"}),
                 hide_index=True, use_container_width=True)
    st.caption("Counts describe this artifact's actual coverage, not all SURFRAD stations or the Indian AWS network.")
    with st.expander("Detector configuration and frozen calibration"):
        st.json(bundle["detector"])
    with st.expander("Recorded evaluation metrics and limitations"):
        st.json(bundle["metrics"])
    with st.expander("Data provenance and admission evidence"):
        found = False
        for name in ("provenance", "data_provenance", "training_manifest"):
            if name in bundle:
                st.json(bundle[name])
                found = True
        if not found:
            st.info("No provenance summary was included in this artifact directory. Inspect the retained "
                    "per-record lineage and the training run's raw-admission audit.")
        st.caption("The dashboard checks artifact compatibility; it does not rerun immutable raw-file "
                   "reconstruction. That verification belongs to the dataset admission step.")
