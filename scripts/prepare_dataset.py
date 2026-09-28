"""Master dataset-preparation pipeline.

Reads raw downloads, produces the three training artifacts that get uploaded
to Kaggle:

  data/processed/train_clean.parquet    (cleaned, chronological 2018-2022)
  data/processed/val_labeled.parquet    (cleaned + injected faults, 2023)
  data/processed/test_labeled.parquet   (cleaned + injected faults, 2024-2025)
  data/processed/injection_events.parquet   (ground-truth fault events)
  data/processed/station_stats.json     (train-only climatology/robust stats)
  data/processed/splits.json            (station holdout + time splits)

Run:  python scripts/prepare_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awsad.data.parse_isd import PARSER_VERSION, parse_all as parse_isd_all  # noqa: E402
from awsad.data.parse_openmeteo import parse_all as parse_om_all     # noqa: E402
from awsad.data.stations import (                                      # noqa: E402
    DEFAULT_HOLDOUT_STATIONS,
    as_dicts,
    build_station_catalog,
    load_extended_stations,
    select_holdout_stations,
)
from awsad.preprocessing.anomaly_injection import AnomalyInjector     # noqa: E402
from awsad.preprocessing.features import (                            # noqa: E402
    CORE_CHANNELS, MEASURED_CHANNELS, add_physics_derived,
    dewpoint_to_rh, impute_hourly)
from awsad.preprocessing.qc_rules import compute_qc_flags             # noqa: E402

TRAIN_END = pd.Timestamp("2023-01-01", tz="UTC")
VAL_END = pd.Timestamp("2024-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-02-01", tz="UTC")   # ISD files extend into 2025/2026

RANDOM_SEED = 42
HOLDOUT_FRACTION = 0.20
MIN_GROUP_ROWS_FOR_TRAINING = 2_000

SOURCE_ROLES = {
    "noaa_isd": (
        "primary_observation; AWS-compatible surface-station proxy; "
        "RH derived from NOAA dewpoint"
    ),
    "openmeteo": "reference_reanalysis; not physical AWS sensor ground truth",
    "nab": "generic univariate anomaly benchmark; not AWS weather labels",
}


def _read_any(f: Path) -> pd.DataFrame:
    if f.suffix == ".parquet":
        return pd.read_parquet(f)
    return pd.read_csv(f, parse_dates=["timestamp"])


def _interim_files(d: Path, stem: str) -> list[Path]:
    return sorted(d.glob(f"{stem}_*.parquet")) or sorted(d.glob(f"{stem}_*.csv.gz"))


def load_interim(root: Path) -> pd.DataFrame:
    frames = []
    for f in _interim_files(root / "data/interim/isd", "isd"):
        d = _read_any(f)
        d["source"] = "noaa_isd"
        d["source_role"] = "primary_observation"
        d["rh_source"] = "derived_from_dewpoint"
        d["pressure_source"] = "noaa_sea_level_pressure"
        # ISD has no RH/radiation -> derive RH from Magnus, radiation unavailable
        if "relative_humidity_pct" not in d.columns:
            d["relative_humidity_pct"] = dewpoint_to_rh(d["temperature_c"], d["dewpoint_c"])
        frames.append(d)
    for f in _interim_files(root / "data/interim/openmeteo", "openmeteo"):
        d = _read_any(f)
        d["source"] = "openmeteo"
        d["source_role"] = "reference_reanalysis"
        d["rh_source"] = "direct_reanalysis"
        d["pressure_source"] = "reanalysis_pressure_msl"
        frames.append(d)
    if not frames:
        raise SystemExit("No interim data found. Run the downloaders + parsers first.")
    return pd.concat(frames, ignore_index=True)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly resample + drop QC-failed raw points (they are real faults) +
    physics-derived channels from the three measured ones."""
    # Preserve duplicate ISD reports until field-wise hourly aggregation; a
    # METAR row can be missing pressure while the synoptic row at the same
    # timestamp contains the valid value.
    df = df.sort_values("timestamp")
    df = impute_hourly(df, [c for c in CORE_CHANNELS if c in df], max_gap=6)
    flags = compute_qc_flags(df)
    hard = [c for c in flags.columns if c.startswith(("range_", "consistency_"))]
    if hard:
        bad = flags[hard].any(axis=1)
        for c in [c for c in CORE_CHANNELS if c in df.columns]:
            df.loc[bad, c] = np.nan
    df = add_physics_derived(df)
    return df


def _train_stats(frame: pd.DataFrame, train_mask: pd.Series) -> dict:
    """Return finite, train-only robust statistics for one station stream."""
    out = {}
    for c in CORE_CHANNELS:
        if c not in frame.columns:
            continue
        tr = pd.to_numeric(frame.loc[train_mask, c], errors="coerce").dropna()
        if tr.empty:
            continue
        med = float(tr.median())
        iqr = float(tr.quantile(.75) - tr.quantile(.25))
        if np.isfinite(med) and np.isfinite(iqr):
            out[c] = {"median": med, "iqr": max(iqr, 1e-3)}
    return out


def main() -> None:
    root = ROOT
    (root / "data/interim/isd").mkdir(parents=True, exist_ok=True)
    (root / "data/interim/openmeteo").mkdir(parents=True, exist_ok=True)

    isd_interim = root / "data/interim/isd"
    isd_files = list(isd_interim.glob("isd_*"))
    marker = isd_interim / ".parser_version"
    needs_reparse = (not isd_files or not marker.exists() or
                     marker.read_text(encoding="utf-8").strip() != PARSER_VERSION)
    if needs_reparse:
        history = root / "data/raw/noaa_isd/isd-history.csv"
        catalog = load_extended_stations(history, end_year=2023) if history.exists() else None
        parse_isd_all(root / "data/raw/noaa_isd", isd_interim, stations=catalog)
        marker.write_text(PARSER_VERSION, encoding="utf-8")
    else:
        print(f"using cached interim ISD ({len(isd_files)} stations)")
    if not list((root / "data/interim/openmeteo").glob("openmeteo_*")):
        parse_om_all(root / "data/raw/openmeteo", root / "data/interim/openmeteo")
    else:
        print("using cached interim openmeteo")

    print("loading interim data ...")
    df = load_interim(root)
    print(f"  combined rows: {len(df):,}")

    rng = np.random.default_rng(RANDOM_SEED)
    injector = AnomalyInjector(rng=rng, target_fraction=0.04)

    train_parts, val_parts, test_parts, events_parts = [], [], [], []
    stats: dict = {}
    group_counts: dict[str, dict[str, int]] = {}

    for (sid, src), frame in df.groupby(["station_id", "source"], sort=True):
        frame = clean_frame(frame)
        # climatology & robust stats from TRAIN slice only (no leakage)
        train_mask = frame["timestamp"] < TRAIN_END
        group = f"{sid}|{src}"
        stats[group] = _train_stats(frame, train_mask)

        clean = frame[train_mask]
        val = frame[(frame.timestamp >= TRAIN_END) & (frame.timestamp < VAL_END)]
        test = frame[(frame.timestamp >= VAL_END) & (frame.timestamp < TEST_END)]
        group_counts[group] = {"train": int(len(clean)), "val": int(len(val)),
                               "test": int(len(test))}

        # injection ONLY into measured channels (T/P/RH); physics-derived
        # channels are recomputed afterwards so consistency holds by construction
        inj_cols = [c for c in MEASURED_CHANNELS if c in frame.columns]
        for part, label in ((val, "val"), (test, "test")):
            if len(part) < 500:
                continue
            corrupted, labels, ev = injector.inject(part.reset_index(drop=True), inj_cols)
            # recompute derived channels post-corruption (drop old derived first)
            derived = ["dewpoint_c", "td_spread", "es_hpa", "vpd_hpa", "press_tendency_3h"]
            corrupted = corrupted.drop(columns=[c for c in derived if c in corrupted])
            corrupted = add_physics_derived(corrupted)
            corrupted["label"] = labels.to_numpy()
            # Keep the event table schema even when a sparse/degenerate group
            # produces no valid fault event.
            ev = ev.copy()
            ev["station_id"] = sid
            ev["source"] = src
            ev["split"] = label
            (val_parts if label == "val" else test_parts).append(corrupted)
            events_parts.append(ev)
        train_parts.append(clean)
        print(f"  {sid} {src:10s} train={len(clean):6d} val={len(val):6d} test={len(test):6d}")

    if not train_parts or not val_parts or not test_parts:
        raise SystemExit("Insufficient processed data: at least one train, val and test group is required.")

    out = root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    events = (pd.concat(events_parts, ignore_index=True)
              if events_parts else pd.DataFrame(
                  columns=["fault", "start", "end", "channels", "station_id", "source", "split"]))

    for name, d in (("train_clean", train), ("val_labeled", val), ("test_labeled", test)):
        d.to_parquet(out / f"{name}.parquet", index=False)
        d.to_csv(out / f"{name}.csv.gz", index=False, compression="gzip")
    events.to_parquet(out / "injection_events.parquet", index=False)

    observed_station_ids = sorted(df["station_id"].astype(str).unique())
    station_catalog = build_station_catalog(
        observed_station_ids, root / "data/raw/noaa_isd/isd-history.csv")
    # Select holdouts only from streams with complete chronological coverage;
    # this keeps the protocol honest when a partial download is supplied.
    eligible = {
        group.split("|", 1)[0]
        for group, counts in group_counts.items()
        if group.endswith("|noaa_isd")
        and all(counts[k] >= MIN_GROUP_ROWS_FOR_TRAINING
                for k in ("train", "val", "test"))
    }
    if not eligible:
        eligible = {
            group.split("|", 1)[0]
            for group, counts in group_counts.items()
            if all(counts[k] >= MIN_GROUP_ROWS_FOR_TRAINING
                   for k in ("train", "val", "test"))
        }
    holdout_stations = select_holdout_stations(
        eligible, fraction=HOLDOUT_FRACTION,
        preferred=DEFAULT_HOLDOUT_STATIONS, seed=RANDOM_SEED)

    catalog_payload = {
        "version": 1,
        "station_count": len(station_catalog),
        "stations": as_dicts(station_catalog),
    }
    (out / "station_catalog.json").write_text(
        json.dumps(catalog_payload, indent=2, allow_nan=False), encoding="utf-8")
    split_payload = {
        "version": 2,
        "train_end": str(TRAIN_END), "val_end": str(VAL_END), "test_end": str(TEST_END),
        "holdout_stations": holdout_stations,
        "holdout_fraction": HOLDOUT_FRACTION,
        "holdout_seed": RANDOM_SEED,
        "holdout_selection": "stable_sha256_rank_with_legacy_preferred_ids",
        "stations": observed_station_ids,
        "station_ids": observed_station_ids,
        "station_count": len(observed_station_ids),
        "eligible_station_ids": sorted(eligible),
        "groups": sorted(group_counts),
        "group_count": len(group_counts),
        "group_counts": group_counts,
        "catalog_file": "station_catalog.json",
        "parser_version": PARSER_VERSION,
        "source_roles": SOURCE_ROLES,
    }
    (out / "station_stats.json").write_text(
        json.dumps(stats, indent=1, allow_nan=False), encoding="utf-8")
    (out / "splits.json").write_text(
        json.dumps(split_payload, indent=2, allow_nan=False), encoding="utf-8")
    (out / "source_metadata.json").write_text(
        json.dumps({
            "status": "local_processed_build",
            "parser_version": PARSER_VERSION,
            "source_roles": SOURCE_ROLES,
            "rh_policy": (
                "NOAA ISD RH is derived from TMP+DEW; "
                "Open-Meteo RH is reanalysis"
            ),
            "pressure_policy": (
                "NOAA ISD SLP and Open-Meteo pressure_msl are sea-level pressure"
            ),
        }, indent=2, allow_nan=False), encoding="utf-8")

    print(f"\nSaved -> {out}")
    for name, d in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:5s}: {len(d):>9,} rows "
              f"({d.memory_usage(deep=True).sum()/1e6:.0f} MB in RAM)")
    print(f"  stations: {len(observed_station_ids)}; groups: {len(group_counts)}; "
          f"holdouts: {len(holdout_stations)}")
    print(f"  injected events: {len(events):,}; "
          f"labeled anomaly rate: {val['label'].mean():.3%} val, {test['label'].mean():.3%} test")


if __name__ == "__main__":
    main()
