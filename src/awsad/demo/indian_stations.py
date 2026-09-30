"""Traceable Indian ISD baselines and explicitly synthetic demonstration series.

This module intentionally does not use the legacy ISD parser/injector or a
trained detector. Native reports, duplicates, raw fields and provider QC remain
unchanged. Calculated RH is not a measured humidity channel, and ISD SLP is not
station pressure. Every artifact here is excluded from real training/evaluation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
ARTIFACT_KIND = "indian_station_demo"
DATASET_POLICY = "demo_only_not_real_training"
DEFAULT_STATION_IDS = (
    "42182099999",  # SAFDARJUNG
    "43057099999",  # BOMBAY / COLABA
    "43279099999",  # CHENNAI INTL
    "42809099999",  # NETAJI SUBHASH CHANDRA BOSE INTL
    "42410099999",  # GUWAHATI INTL / LOKPRIYA GOPINATH BORDOLOI INTL
    "42348099999",  # JAIPUR
)
CHANNELS = {
    "baseline_temperature_c": "synthetic_temperature_c",
    "derived_relative_humidity_pct": "synthetic_relative_humidity_pct",
    "baseline_sea_level_pressure_hpa": "synthetic_sea_level_pressure_hpa",
}
CHANNEL_LABELS = {
    "baseline_temperature_c": "Temperature (°C)",
    "derived_relative_humidity_pct": "Derived relative humidity (%)",
    "baseline_sea_level_pressure_hpa": "Sea-level pressure (hPa)",
}
SCENARIOS = ("baseline", "spike", "drift", "stuck", "dropout")
SOURCE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"
HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
RH_FORMULA = "100 * exp(17.625*Td/(243.04+Td) - 17.625*T/(243.04+T))"
_RAW_CHANNELS = {
    "TMP": ("baseline_temperature_c", 9999, "degC"),
    "DEW": ("baseline_dewpoint_c", 9999, "degC"),
    "SLP": ("baseline_sea_level_pressure_hpa", 99999, "hPa"),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_reference(path: Path, relative_to: Path) -> str:
    """Prefer portable workspace paths, retaining absolute paths for other input."""
    try:
        return path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _decode_numeric_field(raw: pd.Series, sentinel: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    parts = raw.str.split(",")
    tokens = parts.str[0].fillna("")
    codes = parts.str[1].fillna("")
    # A nonnumeric token is retained verbatim but is never turned into a value.
    encoded = pd.to_numeric(tokens.where(tokens.str.fullmatch(r"[+-]?\d+")), errors="coerce")
    values = (encoded / 10.0).where(encoded != sentinel).astype(float)
    return values, tokens, codes


def parse_native_station(
    path: str | Path, station_id: str, *, project_root: str | Path | None = None,
) -> pd.DataFrame:
    """Decode native CSV records without QC filtering, sorting, deduping or filling.

    ``source_row`` is the one-based CSV data-record ordinal, excluding the header;
    it is not a physical text line number (quoted reports can span lines).
    All original fields are stored verbatim in ``raw_<provider column>`` columns.
    """
    path = Path(path)
    root = Path(project_root) if project_root is not None else Path.cwd()
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False,
                      encoding="utf-8-sig", low_memory=False)
    required = {"STATION", "DATE", "SOURCE", "NAME", "LATITUDE", "LONGITUDE",
                "ELEVATION", "REPORT_TYPE", "TMP", "DEW", "SLP"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: required native ISD columns missing: {', '.join(sorted(missing))}.")
    if raw.empty:
        raise ValueError(f"{path}: no native records; choose a nonempty station-year file.")
    if not raw["STATION"].eq(str(station_id)).all():
        raise ValueError(f"{path}: raw STATION does not match requested station {station_id}.")
    times = pd.to_datetime(raw["DATE"], utc=True, errors="coerce")
    if times.isna().any():
        bad = (np.flatnonzero(times.isna().to_numpy()) + 1).tolist()[:5]
        raise ValueError(f"{path}: invalid DATE at native data-record ordinals {bad}; preserve and inspect the raw file.")

    # Preserve every provider field, including raw report strings and QC attributes.
    frame = raw.rename(columns=lambda name: f"raw_{name}").copy()
    frame["timestamp"] = times
    frame["station_id"] = str(station_id)
    frame["source"] = "noaa_isd_global_hourly"
    frame["source_file"] = _file_reference(path, root)
    frame["source_file_sha256"] = sha256_file(path)
    frame["source_row"] = np.arange(1, len(raw) + 1, dtype=np.int64)
    frame["baseline_record_id"] = frame["source_file_sha256"] + ":" + frame["source_row"].astype(str)
    frame["source_url"] = f"{SOURCE_URL}/{path.parent.name}/{path.name}"
    frame["source_retrieved_at_utc"] = pd.NA  # Unknown for these existing local archives.
    frame["report_type"] = raw["REPORT_TYPE"]
    for field, (column, sentinel, _) in _RAW_CHANNELS.items():
        values, tokens, codes = _decode_numeric_field(raw[field], sentinel)
        frame[column] = values
        frame[f"{column}__raw_value"] = tokens
        frame[f"{column}__raw_qc"] = codes
    temperature = frame["baseline_temperature_c"].to_numpy(dtype=float)
    dewpoint = frame["baseline_dewpoint_c"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rh = 100.0 * np.exp(17.625 * dewpoint / (243.04 + dewpoint)
                            - 17.625 * temperature / (243.04 + temperature))
    # Derived arithmetic failures remain missing. Do not clip supersaturated or
    # suspect source reports into an apparently clean measurement.
    rh[~np.isfinite(rh) | ~np.isfinite(temperature) | ~np.isfinite(dewpoint)] = np.nan
    frame["derived_relative_humidity_pct"] = rh
    frame["relative_humidity_origin"] = "derived_from_same_record_TMP_and_DEW_not_measured_RH"
    frame["pressure_kind"] = "sea_level_pressure_not_station_pressure"
    frame["hardware_fault_status"] = "unknown"
    frame["eligible_for_real_training"] = False
    frame["is_synthetic_demo"] = False
    return frame


def _channel_metadata() -> dict[str, dict[str, Any]]:
    result = {
        column: {
            "unit": unit, "raw_field": field, "conversion": "raw numeric token / 10",
            "missing_sentinel": sentinel, "qc_column": f"{column}__raw_qc",
            "origin": "unit_conversion_of_archived_provider_field",
            "qc_policy": "retain every parseable non-sentinel value and original QC code",
            "field_source": "raw_SOURCE; raw_REPORT_TYPE; raw_QUALITY_CONTROL when published",
            "row_lineage": "source_file, source_file_sha256, source_row, raw_DATE",
        }
        for field, (column, sentinel, unit) in _RAW_CHANNELS.items()
    }
    result["baseline_sea_level_pressure_hpa"]["pressure_kind"] = "sea_level_not_station"
    result["derived_relative_humidity_pct"] = {
        "unit": "%", "origin": "derived_not_independently_measured",
        "inputs": ["baseline_temperature_c", "baseline_dewpoint_c"],
        "formula": RH_FORMULA, "clipping": "none", "qc_column": None,
        "quality_evidence": ["baseline_temperature_c__raw_qc", "baseline_dewpoint_c__raw_qc"],
        "qc_note": "Parent QC codes are retained, never assigned as humidity-sensor fault truth.",
    }
    return result


def build_demo_bundle(
    raw_dir: str | Path, output_dir: str | Path, *, year: int = 2024,
    station_ids: tuple[str, ...] | list[str] = DEFAULT_STATION_IDS,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a separate demo bundle from existing originals, verifying Indian metadata.

    Rebuilding a recognized demo directory is allowed. Arbitrary existing output
    directories and paths inside the raw source tree are rejected.
    """
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    if output_dir == raw_dir or output_dir.is_relative_to(raw_dir):
        raise ValueError("Demo output must be outside the immutable raw source directory.")
    for protected in (root / "data/raw", root / "data/processed", root / "data/interim"):
        if output_dir == protected or output_dir.is_relative_to(protected):
            raise ValueError("Demo output must stay separate from raw, processed and interim training data.")
    if output_dir.exists() and any(output_dir.iterdir()):
        existing_manifest = output_dir / "manifest.json"
        try:
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Output directory is not a recognized demo bundle; choose a new demo directory.") from exc
        if existing.get("artifact_kind") != ARTIFACT_KIND:
            raise ValueError("Output directory belongs to another artifact type; choose a separate demo directory.")
    station_ids = tuple(str(sid) for sid in station_ids)
    if not station_ids or len(set(station_ids)) != len(station_ids):
        raise ValueError("Choose at least one station ID, without duplicates.")
    if any(len(sid) != 11 or not sid.isdigit() for sid in station_ids):
        raise ValueError("ISD station IDs must be the 11-digit USAF+WBAN filename key.")
    history_path = raw_dir / "isd-history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing station metadata: {history_path}. Restore the original isd-history.csv.")
    history = pd.read_csv(history_path, dtype=str, keep_default_na=False, na_filter=False)
    required = {"USAF", "WBAN", "STATION NAME", "CTRY", "LAT", "LON", "ELEV(M)"}
    if not required.issubset(history.columns):
        raise ValueError("isd-history.csv is missing required station identity or geographic fields.")
    keys = history["USAF"] + history["WBAN"]
    history_hash = sha256_file(history_path)
    frames, records, sources = [], [], []
    # Validate every input before writing output files.
    for station_id in station_ids:
        metadata_rows = history.loc[keys.eq(station_id)]
        if len(metadata_rows) != 1 or metadata_rows.iloc[0]["CTRY"] != "IN":
            raise ValueError(f"Station {station_id}: require one matching USAF+WBAN metadata row with CTRY=IN.")
        metadata = metadata_rows.iloc[0]
        path = raw_dir / str(year) / f"{station_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing original {path}. Choose an available station-year; no fallback data are generated.")
        frame = parse_native_station(path, station_id, project_root=root)
        if not frame["timestamp"].dt.year.eq(int(year)).all():
            raise ValueError(f"{path}: raw timestamps fall outside requested year {year}.")
        coordinates = pd.to_numeric(metadata[["LAT", "LON", "ELEV(M)"]], errors="coerce")
        if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
            raise ValueError(f"Station {station_id}: missing published coordinates/elevation; do not invent them.")
        record = {
            "station_id": station_id, "name": metadata["STATION NAME"], "country": "IN",
            "latitude": float(coordinates["LAT"]), "longitude": float(coordinates["LON"]),
            "elevation_m": float(coordinates["ELEV(M)"]), "year": year, "rows": len(frame),
            "start_utc": frame["timestamp"].min().isoformat(),
            "end_utc": frame["timestamp"].max().isoformat(),
            "temperature_present": int(frame["baseline_temperature_c"].notna().sum()),
            "dewpoint_present": int(frame["baseline_dewpoint_c"].notna().sum()),
            "rh_derived_present": int(frame["derived_relative_humidity_pct"].notna().sum()),
            "sea_level_pressure_present": int(frame["baseline_sea_level_pressure_hpa"].notna().sum()),
            "duplicate_timestamp_rows": int(frame["timestamp"].duplicated().sum()),
            "source_file": frame["source_file"].iloc[0],
            "source_file_sha256": frame["source_file_sha256"].iloc[0],
            "history_sha256": history_hash,
        }
        source = {
            "station_id": station_id, "path": record["source_file"],
            "sha256": record["source_file_sha256"], "bytes": path.stat().st_size,
            "url": frame["source_url"].iloc[0], "retrieved_at_utc": None,
            "origin_check": "existing local archive; live origin not reverified by this builder",
            "rows": len(frame), "raw_columns": [column[4:] for column in frame if column.startswith("raw_")],
            "qc_counts": {
                field: {str(key): int(value) for key, value in frame[f"{column}__raw_qc"].value_counts(dropna=False).items()}
                for field, (column, _, _) in _RAW_CHANNELS.items()
            },
        }
        frames.append(frame)
        records.append(record)
        sources.append(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "baseline_observations"
    baseline_dir.mkdir(exist_ok=True)
    station_files = []
    for frame, record in zip(frames, records, strict=True):
        destination = baseline_dir / f"{record['station_id']}_{year}.parquet"
        frame.to_parquet(destination, index=False)
        station_files.append({"station_id": record["station_id"],
                              "path": destination.relative_to(output_dir).as_posix(),
                              "sha256": sha256_file(destination), "rows": len(frame)})
    catalog_path = output_dir / "station_catalog.csv"
    pd.DataFrame(records).to_csv(catalog_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION, "artifact_kind": ARTIFACT_KIND,
        "dataset_policy": DATASET_POLICY, "eligible_for_real_training": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "year": year,
        "station_count": len(records), "row_count": sum(record["rows"] for record in records),
        "baseline_origin": "native archived Indian surface-station ISD reports; not independently verified AWS installations",
        "scenario_origin": "explicitly synthetic demo copies and deterministic transformations; never measured observations",
        "purpose": "user-requested separate Indian-station dashboard demonstration; no training, selection or benchmark use",
        "hardware_fault_status": "unknown", "provider_qc_is_fault_truth": False,
        "native_record_policy": "all records in source order, timestamps/duplicates/missing values preserved; no resampling or filling",
        "source_row_definition": "one-based CSV data-record ordinal excluding header; not physical line number",
        "source_metadata": {"path": _file_reference(history_path, root), "sha256": history_hash,
                            "url": HISTORY_URL, "retrieved_at_utc": None,
                            "verification": "local USAF+WBAN identity, CTRY=IN and published coordinates checked"},
        "attribution": "NOAA National Centers for Environmental Information, Integrated Surface Database (Global Hourly).",
        "rights_note": "Existing local archive only; no new licensing or redistribution-rights verification claimed.",
        "channels": _channel_metadata(), "raw_sources": sources, "station_files": station_files,
        "catalog_file": {"path": catalog_path.name, "sha256": sha256_file(catalog_path)},
        "scenario_version": 1, "scenarios": list(SCENARIOS),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _checked_bundle_file(root: Path, file_spec: dict[str, Any]) -> Path:
    relative = Path(file_spec.get("path", ""))
    path = (root / relative).resolve()
    if relative.is_absolute() or not relative.parts or not path.is_relative_to(root):
        raise ValueError("Demo manifest file paths must stay inside the demo bundle.")
    if not path.is_file():
        raise FileNotFoundError(f"Missing demo file {path}; rebuild with scripts/build_indian_demo.py.")
    if sha256_file(path) != file_spec.get("sha256"):
        raise ValueError(f"Demo file hash mismatch: {path.name}. Rebuild from the original archive before use.")
    return path


def load_demo_bundle(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No Indian demo manifest at {manifest_path}; run scripts/build_indian_demo.py.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("artifact_kind") != ARTIFACT_KIND
            or manifest.get("dataset_policy") != DATASET_POLICY
            or manifest.get("eligible_for_real_training") is not False):
        raise ValueError("This is not an isolated Indian-station demo bundle. Choose the output of scripts/build_indian_demo.py.")
    catalog_path = _checked_bundle_file(root, manifest.get("catalog_file", {}))
    catalog = pd.read_csv(catalog_path, dtype={"station_id": str}, keep_default_na=False)
    required = {"station_id", "name", "country", "latitude", "longitude", "rows"}
    if not required.issubset(catalog.columns) or catalog.empty or not catalog["country"].eq("IN").all():
        raise ValueError("Demo station catalog must contain Indian station metadata and counts.")
    specs = manifest.get("station_files", [])
    ids = [spec.get("station_id") for spec in specs]
    if (catalog["station_id"].duplicated().any() or len(set(ids)) != len(ids)
            or set(ids) != set(catalog["station_id"])
            or len(catalog) != manifest.get("station_count")
            or int(catalog["rows"].sum()) != manifest.get("row_count")):
        raise ValueError("Demo station manifest/catalog mismatch; rebuild the demo bundle.")
    return {"root": root, "manifest": manifest, "catalog": catalog}


def read_demo_station(bundle: dict[str, Any], station_id: str) -> pd.DataFrame:
    station_id = str(station_id)
    specs = [spec for spec in bundle["manifest"]["station_files"] if spec["station_id"] == station_id]
    if len(specs) != 1:
        raise ValueError(f"Station {station_id} is not in this demo bundle. Select a station from its catalog.")
    path = _checked_bundle_file(Path(bundle["root"]).resolve(), specs[0])
    frame = pd.read_parquet(path)
    required = {"timestamp", "station_id", "source_row", "source_file_sha256",
                "hardware_fault_status", "eligible_for_real_training", *CHANNELS}
    if not required.issubset(frame.columns) or len(frame) != specs[0]["rows"]:
        raise ValueError(f"{path.name}: invalid native baseline schema or record count; rebuild the demo bundle.")
    if (not frame["station_id"].eq(station_id).all()
            or not frame["eligible_for_real_training"].eq(False).all()
            or not frame["hardware_fault_status"].eq("unknown").all()):
        raise ValueError(f"{path.name}: station identity or demo-only eligibility markers do not match.")
    return frame


def simulate_scenario(
    frame: pd.DataFrame, channel: str, scenario: str = "baseline",
    start: Any = None, duration_hours: float = 6.0, magnitude: float = 5.0,
) -> pd.DataFrame:
    """Return demo-generated series beside unchanged native baseline columns.

    The interval is [start, start + duration). Spike changes the first eligible
    timestamp (all finite duplicate reports there); drift adds elapsed-time / 
    duration * magnitude; stuck copies the first finite interval value; dropout
    removes finite interval values. Original missing readings stay missing.
    ``scenario_applied`` marks the selected report interval; ``injected_demo_change``
    marks actual value changes and is never a confirmed hardware-fault label.
    No random sampling or artificially generated timestamps are used.
    """
    if channel not in CHANNELS:
        raise ValueError(f"Unknown demo channel {channel!r}; choose one of {', '.join(CHANNELS)}.")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose one of {', '.join(SCENARIOS)}.")
    required = {"timestamp", *CHANNELS}
    if not required.issubset(frame.columns):
        raise ValueError("Scenario input must contain timestamps and all three explicit baseline channels.")
    if frame.empty:
        raise ValueError("No native reports in this window. Choose a window with archived reports.")
    if "station_id" in frame and frame["station_id"].nunique(dropna=False) != 1:
        raise ValueError("Run one station at a time; mixed station scenarios are not supported.")
    if not np.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError("Scenario duration_hours must be finite and greater than zero.")
    if not np.isfinite(magnitude):
        raise ValueError("Scenario magnitude must be a finite number in the displayed channel unit.")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Scenario input contains invalid timestamps; inspect the baseline source.")
    start_time = timestamps.min() if start is None else pd.to_datetime(start, utc=True, errors="coerce")
    if pd.isna(start_time):
        raise ValueError("Scenario start must be a valid UTC timestamp.")
    end_time = start_time + pd.Timedelta(hours=float(duration_hours))
    output = frame.copy(deep=True)
    for baseline, synthetic in CHANNELS.items():
        output[synthetic] = pd.to_numeric(frame[baseline], errors="coerce").to_numpy(dtype=float)
    baseline_values = output[CHANNELS[channel]].to_numpy(dtype=float, copy=True)
    synthetic_values = baseline_values.copy()
    in_interval = ((timestamps >= start_time) & (timestamps < end_time)).to_numpy(dtype=bool)
    finite = np.isfinite(baseline_values)
    eligible = in_interval & finite
    applied = np.zeros(len(frame), dtype=bool)
    if scenario != "baseline":
        if not eligible.any():
            raise ValueError("No finite baseline values for this channel in the selected interval. Choose another channel or interval; missing observations are never filled.")
        applied = in_interval.copy()
        first_time = timestamps.iloc[np.flatnonzero(eligible)].min()
        if scenario == "spike":
            applied &= timestamps.eq(first_time).to_numpy(dtype=bool)
            synthetic_values[applied & finite] += magnitude
        elif scenario == "drift":
            fraction = ((timestamps - start_time).dt.total_seconds() / (duration_hours * 3600)).to_numpy(dtype=float)
            synthetic_values[eligible] += magnitude * fraction[eligible]
        elif scenario == "stuck":
            first_position = np.flatnonzero(eligible & timestamps.eq(first_time).to_numpy(dtype=bool))[0]
            synthetic_values[eligible] = baseline_values[first_position]
        elif scenario == "dropout":
            synthetic_values[eligible] = np.nan
    output[CHANNELS[channel]] = synthetic_values
    equal_values = (synthetic_values == baseline_values) | (np.isnan(synthetic_values) & np.isnan(baseline_values))
    output["is_synthetic_demo"] = True
    output["scenario_applied"] = applied
    output["injected_demo_change"] = ~equal_values
    output["scenario_type"] = scenario
    output["scenario_channel"] = channel
    output["scenario_start_utc"] = start_time
    output["scenario_duration_hours"] = float(duration_hours)
    output["scenario_magnitude"] = float(magnitude)
    output["scenario_version"] = 1
    output["scenario_data_origin"] = "demo_generated_copy_with_explicit_synthetic_transform"
    output["hardware_fault_status"] = "unknown"
    output["eligible_for_real_training"] = False
    return output
