"""Read-only views of real-observation detector artifacts.

Chart segmentation changes presentation only: it never inserts or fills a row.
Software fixtures for these helpers belong in tests, never in replay artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

CHANNELS = ("temperature_c", "relative_humidity_pct", "pressure_hpa")
CHANNEL_LABELS = {
    "temperature_c": "Temperature (°C)",
    "relative_humidity_pct": "Measured relative humidity (%)",
    "pressure_hpa": "Station pressure (hPa)",
}
REQUIRED_COLUMNS = {
    "timestamp", "group", "station_id", "source", "observation_id",
    "raw_file_sha256", "raw_row_number", *CHANNELS,
    "anomaly_score", "is_candidate", "reason_codes",
}


def load_bundle(directory: str | Path) -> dict:
    """Reject legacy artifacts and inspect only parquet metadata at startup.

    This checks the replay contract, not the immutable raw-file admission audit
    performed by the training pipeline. The audit remains separately visible.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Artifacts directory does not exist: {root}")
    metadata = {}
    for name in ("metrics", "detector", "provenance", "data_provenance", "training_manifest"):
        path = root / f"{name}.json"
        if path.is_file():
            try:
                metadata[name] = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError(f"Cannot read {path.name}: {exc}") from exc
    for name in ("metrics", "detector"):
        if metadata.get(name, {}).get("dataset_policy") != "real_observations_only":
            raise ValueError(
                "This dashboard requires minute detector artifacts declaring "
                "dataset_policy='real_observations_only' in metrics.json and detector.json. "
                "Legacy injected-data and hourly forecasting bundles are not compatible."
            )
    files = sorted((root / "scored_observations").glob("*.parquet"))
    if (root / "scored_observations.parquet").is_file():
        files = [root / "scored_observations.parquet"]
    if not files:
        raise ValueError("No scored_observations.parquet or scored_observations/*.parquet was found.")
    catalog_rows = []
    for path in files:
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"{path.name} lacks replay fields: {', '.join(sorted(missing))}")
        group_index, time_index = columns.index("group"), columns.index("timestamp")
        shard_rows = []
        for index in range(parquet.num_row_groups):
            row_group = parquet.metadata.row_group(index)
            group_stats = row_group.column(group_index).statistics
            time_stats = row_group.column(time_index).statistics
            if (group_stats is None or time_stats is None or not group_stats.has_min_max
                    or not time_stats.has_min_max or group_stats.min != group_stats.max):
                shard_rows = []
                break
            shard_rows.append({"group": str(group_stats.min), "records": row_group.num_rows,
                               "first": time_stats.min, "last": time_stats.max})
        if not shard_rows:
            # Handles combined or old unpartitioned real detector exports. Read
            # only identifiers/times; wide raw provenance stays on disk.
            compact = parquet.read(columns=["group", "timestamp"]).to_pandas()
            shard_rows = compact.groupby("group", sort=True).agg(
                records=("timestamp", "size"), first=("timestamp", "min"),
                last=("timestamp", "max")).reset_index().to_dict("records")
        catalog_rows.extend(shard_rows)
    if not catalog_rows:
        raise ValueError("The scored observation files contain no records.")
    catalog = pd.DataFrame(catalog_rows).groupby("group", as_index=False).agg(
        records=("records", "sum"), first=("first", "min"), last=("last", "max"))
    for column in ("first", "last"):
        catalog[column] = pd.to_datetime(catalog[column], utc=True)
    events_path = root / "candidate_events.csv"
    events = pd.DataFrame()
    if events_path.is_file() and events_path.stat().st_size:
        try:
            events = pd.read_csv(events_path, dtype={"group": "string", "event_id": "string"})
        except pd.errors.EmptyDataError:
            pass
        for column in ("start", "end"):
            if column in events:
                events[column] = pd.to_datetime(events[column], utc=True)
    return {"root": root, "files": files, "catalog": catalog, "events": events, **metadata}


def read_observations(bundle: dict, group: str, start, end) -> pd.DataFrame:
    """Read an inclusive UTC window without resampling, filling or QC masking."""
    dataset = ds.dataset([str(path) for path in bundle["files"]], format="parquet")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    frame = dataset.to_table(filter=(
        (ds.field("group") == group) & (ds.field("timestamp") >= start.to_pydatetime())
        & (ds.field("timestamp") <= end.to_pydatetime())
    )).to_pandas()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values(["timestamp", "observation_id"], kind="stable").reset_index(drop=True)


def boolean_flags(values: pd.Series) -> pd.Series:
    """Only explicit true values are candidates; unknown remains unscored."""
    return values.eq(True).fillna(False)  # noqa: E712


def chart_series(frame: pd.DataFrame, channel: str, *, include_long_horizon: bool = False,
                 cadence_seconds: int = 60) -> pd.DataFrame:
    """Give each contiguous observed/predicted line its own chart segment.

    Missing channel values and missing timestamps both break lines. Values and
    timestamps in the returned chart are always taken from existing rows.
    """
    choices = [(channel, "Observed"), (f"{channel}__prediction", "Model: 1-minute horizon")]
    if include_long_horizon:
        choices.append((f"{channel}__prediction_60m", "Model: 60-minute horizon"))
    result = []
    gap = frame["timestamp"].diff().dt.total_seconds().ne(cadence_seconds)
    for column, label in choices:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        present = values.notna() & np.isfinite(values)
        segments = (gap | ~present | ~present.shift(1, fill_value=False)).cumsum()
        view = pd.DataFrame({"timestamp": frame["timestamp"], "value": values,
                             "series": label, "segment": label + ":" + segments.astype(str)})
        result.append(view.loc[present])
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame(
        columns=["timestamp", "value", "series", "segment"])


def missing_intervals(frame: pd.DataFrame, *, cadence_seconds: int = 60) -> pd.DataFrame:
    """Describe internal archived-row gaps, not a diagnosed telemetry fault."""
    columns = ["last_observed_utc", "next_observed_utc", "unreported_minute_slots"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    timestamps = frame["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    delta = timestamps.diff().dt.total_seconds()
    selected = delta.gt(cadence_seconds)
    return pd.DataFrame({
        columns[0]: timestamps.shift()[selected], columns[1]: timestamps[selected],
        columns[2]: (np.ceil(delta[selected] / cadence_seconds) - 1).astype(int),
    }).reset_index(drop=True)


def window_summary(frame: pd.DataFrame) -> dict:
    candidates = int(boolean_flags(frame["is_candidate"]).sum())
    scored = int(frame["anomaly_score"].notna().sum())
    missing_values = int(frame[list(CHANNELS)].isna().sum().sum())
    gaps = missing_intervals(frame)
    return {"records": len(frame), "candidates": candidates, "scored": scored,
            "missing_values": missing_values,
            "unreported_slots": int(gaps["unreported_minute_slots"].sum())}


def review_export(bundle: dict, group: str) -> pd.DataFrame:
    """Keep review status unknown unless supplied by a separate review artifact."""
    path = bundle["root"] / "review_template.csv"
    try:
        reviews = pd.read_csv(path) if path.is_file() and path.stat().st_size else bundle["events"].copy()
    except pd.errors.EmptyDataError:
        reviews = bundle["events"].copy()
    if "group" in reviews:
        reviews = reviews.loc[reviews["group"].eq(group)].copy()
    if "review_status" not in reviews:
        reviews["review_status"] = "unknown"
    reviews["review_status"] = reviews["review_status"].fillna("unknown")
    for column in ("reviewer", "evidence_url_or_path", "evidence_notes"):
        if column not in reviews:
            reviews[column] = ""
    return reviews
