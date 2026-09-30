"""SkyGuard real replay and separate Indian scenarios. Run with streamlit."""
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
sys.path.insert(0, str(ROOT / "src"))
from app.real_dashboard import (CHANNELS, CHANNEL_LABELS, boolean_flags, chart_series,
                                load_bundle, missing_intervals, read_observations,
                                observed_record_columns, review_export, signal_evidence, window_summary)
from app.theme import palette, inject_css, configure_chart

st.set_page_config(page_title="SkyGuard · Station explorer", page_icon="🌦", layout="wide")
alt.data_transformers.disable_max_rows()
PALETTE = palette()
inject_css(PALETTE)


@st.cache_resource(show_spinner=False)
def cached_bundle(directory: str, revision: int):
    return load_bundle(directory)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_window(directory: str, revision: int, group: str, start: str, end: str):
    return read_observations(cached_bundle(directory, revision), group, start, end)


def render_channel(frame: pd.DataFrame, channel: str, long_horizon: bool, p: dict):
    st.markdown(f"**{CHANNEL_LABELS[channel]}**")
    lines = chart_series(frame, channel, include_long_horizon=long_horizon)
    if lines.empty:
        st.info("No measured values or model predictions are available in this window.")
        return
    domain = ["Observed", "Model: 1-minute horizon", "Model: 60-minute horizon"]
    color = alt.Color("series:N", scale=alt.Scale(domain=domain, range=[p["ink"], p["accent"], p["accent2"]]),
                      legend=alt.Legend(title=None, orient="top"))
    dash = alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=[[1, 0], [5, 3], [1, 3]]), legend=None)
    y = alt.Y("value:Q", title=None, scale=alt.Scale(zero=False))
    x = alt.X("timestamp:T", title="UTC", scale=alt.Scale(type="utc"), axis=alt.Axis(format="%d %b %H:%M"))
    base = alt.Chart(lines)
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["timestamp"], empty=False)
    gradient = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                            stops=[alt.GradientStop(color=p["area_bottom"], offset=0),
                                   alt.GradientStop(color=p["area_top"], offset=1)])
    area = (base.transform_filter("datum.series === 'Observed'")
            .mark_area(interpolate="monotone", line=False, color=gradient)
            .encode(x=x, y=y, detail="segment:N"))
    line = base.mark_line(strokeWidth=2, interpolate="monotone").encode(
        x=x, y=y, color=color, strokeDash=dash, detail="segment:N")
    hover = base.mark_point(size=66, filled=True).encode(
        x=x, y=y, color=color, opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("timestamp_utc:N", title="Time"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("value:Q", title="Value", format=".3f")])
    rule = base.mark_rule(color=p["faint"], strokeWidth=1).encode(
        x=x, opacity=alt.condition(nearest, alt.value(0.45), alt.value(0))).add_params(nearest)
    layers = [area, line, rule, hover]
    alert_column = f"{channel}__alert"
    if alert_column in frame:
        marker_columns = [name for name in ("timestamp", channel, f"{channel}__reason_codes") if name in frame]
        flagged = frame.loc[boolean_flags(frame[alert_column]) & frame[channel].notna(), marker_columns].copy()
        if not flagged.empty:
            flagged["timestamp_utc"] = flagged["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            tooltip = [alt.Tooltip("timestamp_utc:N", title="Time"), alt.Tooltip(f"{channel}:Q", title="Observed", format=".3f")]
            if f"{channel}__reason_codes" in flagged:
                tooltip.append(alt.Tooltip(f"{channel}__reason_codes:N", title="Candidate reason"))
            layers.append(alt.Chart(flagged).mark_point(color=p["alert"], size=95, filled=False, strokeWidth=1.9).encode(
                x=alt.X("timestamp:T", scale=alt.Scale(type="utc")), y=alt.Y(f"{channel}:Q"), tooltip=tooltip))
    st.altair_chart(configure_chart(alt.layer(*layers), p, height=250), use_container_width=True)


st.sidebar.title("SkyGuard AI")
dashboard_page = st.sidebar.selectbox(
    "Dashboard page", ["USA · Real observations", "India · Synthetic scenarios"],
    key="dashboard_page",
)
if dashboard_page == "India · Synthetic scenarios":
    from app.indian_dashboard import render_indian_dashboard

    render_indian_dashboard(ROOT)
    st.stop()

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
group = st.sidebar.selectbox("Station | observation source", catalog["group"].tolist(), key="station_group")
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
columns[0].metric("Observed rows", f"{summary['records']:,}",
                  help="Existing one-minute station records in this window. Each row has up to three measured sensor values.")
columns[1].metric("Candidate rows", f"{summary['candidates']:,}",
                  help="A detector threshold was crossed; this is not a confirmed hardware fault.")
columns[2].metric("Missing readings", f"{summary['missing_values']:,}",
                  help="Absent values among the three channels in existing archived rows.")
columns[3].metric("Unreported minutes", f"{summary['unreported_slots']:,}",
                  help="Internal timestamp gaps in this window. Archive gaps do not diagnose telemetry failure.")
st.caption(f"{start:%d %b %Y %H:%M} — {end:%d %b %Y %H:%M} UTC · "
           f"{summary['scored']:,} rows have an anomaly score. Missing scores mean no score is available.")
with st.expander("What do these numbers mean?"):
    st.markdown(f"""
These four cards cover **this station and selected time window**.

| Number | Meaning in this view |
|---|---|
| **{summary['records']:,} observed rows** | Existing one-minute records, each with up to three sensor readings. This is not a count of stations or training examples. |
| **{summary['candidates']:,} candidate rows** | Minutes where at least one check crossed its threshold on at least one channel. A minute is counted once even if several checks fire. |
| **{summary['missing_values']:,} missing readings** | Individual missing temperature, humidity or pressure values inside existing rows. One row can contribute up to three. |
| **{summary['unreported_slots']:,} unreported minutes** | Missing minute slots between records in this window. This does not include unobserved time outside the first/last record or diagnose a telemetry failure. |
| **{summary['scored']:,} scored rows** | Rows with at least one available signal score; some other checks may lack the required history. |
| **{int(station['records']):,} archived rows** | The sidebar total for this station across the entire loaded artifact, before filtering to the window. |

An **event** groups consecutive flagged minutes for one channel. Many candidate
rows can belong to one event. Events break at gaps and output boundaries.
""")
    if summary["records"]:
        st.caption(f"This window's candidate fraction is {summary['candidates']:,} / {summary['records']:,} = "
                   f"{100 * summary['candidates'] / summary['records']:.1f}%. "
                   "It is not model accuracy or a false-alarm rate. A window centred on a candidate deliberately contains more alerts.")
    st.markdown("**Reading the plots:** temperature is °C, humidity is %, and pressure is station pressure in hPa. "
                "The solid line is the actual measurement. The terracotta prediction at time t was made using readings "
                "available at least one minute before t. The optional teal prediction uses readings ending "
                "60 minutes before t. The context slider adds viewing time before/after an event; it does not change detection.")
    st.markdown("**Anomaly score:** the largest signal-to-threshold ratio among available checks. "
                "For positive thresholds, 1 is the boundary and a score above 1 means at least one threshold was crossed. "
                "A score of 2 means twice a reference threshold, not a 200% fault probability. "
                "When a threshold is zero, a positive signal uses 2 as an exceedance marker instead of a ratio. "
                "No score means unavailable, not normal.")

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
        st.caption("The solid line is the native measurement; model outputs are separate dashed lines "
                   "(terracotta = 1-minute, teal = 60-minute). Open crimson circles mark candidate readings. "
                   "Gaps break the lines; hover for exact values.")
        for channel in CHANNELS:
            render_channel(frame, channel, long_horizon, PALETTE)
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
        with st.expander("Explain a reading's score and thresholds"):
            default_index = int(flagged.index[0]) if not flagged.empty else int(frame.index[0])
            score_index = st.selectbox("Reading whose score to explain", frame.index.tolist(),
                                      index=frame.index.tolist().index(default_index),
                                      format_func=lambda row: f"{frame.loc[row, 'timestamp']:%Y-%m-%d %H:%M:%S} UTC",
                                      key="score_explanation_row")
            selected_row = frame.loc[score_index]
            evidence = signal_evidence(selected_row, bundle["detector"])
            saved_score = selected_row["anomaly_score"]
            score_text = f"{saved_score:.6g}" if pd.notna(saved_score) else "unavailable"
            st.write(f"Saved row score: **{score_text}**. Hardware-fault status: **unknown**.")
            crossed = evidence.loc[evidence.crossed.eq("Yes")]
            if not crossed.empty:
                strongest = crossed.loc[crossed.threshold_multiple.idxmax()]
                st.write(f"Largest crossed check: {strongest['channel']} — {strongest['check']}.")
                if strongest.zero_threshold_marker:
                    st.write(f"Signal {strongest['signal_value']:.6g} exceeds a zero threshold; score 2 is an exceedance marker.")
                else:
                    st.write(f"{strongest['signal_value']:.6g} / {strongest['threshold']:.6g} = "
                             f"{strongest['threshold_multiple']:.6g} × its reference threshold.")
            st.dataframe(evidence.rename(columns={"signal_value": "Signal value", "threshold": "Threshold",
                         "threshold_multiple": "Threshold multiple", "reference_rows": "Reference rows",
                         "channel": "Channel", "check": "Check", "unit": "Unit", "crossed": "Crossed"})
                         .drop(columns=["reference_group", "zero_threshold_marker"]), hide_index=True, use_container_width=True)
            window = bundle["detector"].get("config", {}).get("sustained_minutes", 30)
            st.caption(f"Forecast errors are absolute differences from actual readings. Sustained error is the absolute "
                       f"mean signed 60-minute forecast error over {window} consecutive minutes. Unchanged-value duration "
                       "is elapsed minutes since the last change, not the number of flagged minutes. Reference rows are "
                       "earlier provider-accepted calibration observations, not verified fault-free examples.")
            if evidence.zero_threshold_marker.any():
                st.caption("Some thresholds are zero: their positive signals use score 2 as a marker, not a ratio.")
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
            observed_columns = observed_record_columns(frame)
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
    metrics = bundle["metrics"]
    total_names = {"native_observations": "Recorded station-minutes", "candidate_minutes": "Flagged station-minutes",
                   "channel_alerts": "Flagged channel-minutes", "candidate_event_proposals": "Candidate intervals",
                   "review_proposals": "Review proposals", "known_hardware_fault_labels": "Confirmed hardware-fault labels"}
    total_rows = [{"Count": title, "Value": metrics[key]} for key, title in total_names.items() if key in metrics]
    if total_rows:
        st.markdown("**Whole-run counts, across all stations**")
        st.dataframe(pd.DataFrame(total_rows), hide_index=True, use_container_width=True)
        st.caption("One station-minute may contain alerts on several channels. Consecutive alerts on one channel form "
                   "an interval. Review proposals also include non-alert samples; all are unknown until independently reviewed.")
    with st.expander("Meaning of evaluation and calibration numbers"):
        st.markdown("**MAE** is the mean absolute forecast error in °C, hPa or humidity percentage points. "
                    "The persistence baseline predicts that the last reading stays unchanged. Lower forecast error does not establish better fault detection.")
        st.markdown("**Provider QC 0** means the provider's checks passed; higher codes report a quality concern. "
                    "Neither confirms a hardware cause. **0 confirmed labels** means none are established, not that no faults occurred. "
                    "**null** recall, false-alarm rate or delay means unavailable, not zero performance.")
        st.markdown("**Threshold** is the frozen boundary learned from earlier reference observations. "
                    "**Reference rows** counts eligible calibration samples. **Quantile** identifies the empirical upper tail used for that boundary. "
                    "The **reference budget** is a calibration design parameter, not a measured accuracy or promised false-alarm rate.")
        reference_budget = bundle["detector"].get("config", {}).get("reference_alert_budget")
        if reference_budget is not None:
            st.caption(f"This detector's reference budget is {reference_budget:g} = {100 * reference_budget:g}%, "
                       "divided across 15 channel/check combinations.")
        st.markdown("**Fit/selection row counts** are sampled training/validation examples per forecaster. "
                    "Dates delimit chronological partitions. Seeds make sampling repeatable. "
                    "SHA-256 values and IDs identify exact files/records; they are not quality scores.")
    with st.expander("Detector configuration and frozen calibration"):
        st.json(bundle["detector"])
    with st.expander("Recorded evaluation metrics and limitations"):
        display_metrics = dict(metrics)
        evaluation = bundle.get("independent_review_evaluation")
        if evaluation is not None:
            display_metrics["real_event_evaluation"] = evaluation
            st.caption("Real-event counts below come from the completed review evaluator, verified against this detector and replay. "
                       "Its decision rows count each channel separately, so three decision rows correspond to one station-minute.")
        else:
            display_metrics.pop("real_event_evaluation", None)
            st.caption("No completed per-observation review evaluation is attached. Initial empty-input placeholders are omitted. "
                       "Independent fault metrics remain unavailable.")
        st.json(display_metrics)
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
