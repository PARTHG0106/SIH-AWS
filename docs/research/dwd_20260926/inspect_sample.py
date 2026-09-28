"""Read downloaded DWD evidence; no dataset generation, filling or labels."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
DOCS = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/dwd_verified_research"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(path):
    result = {"archive": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes()),
              "members": [], "products": [], "metadata": []}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            data = archive.read(member)
            result["members"].append({"name": member.filename, "bytes": len(data), "sha256": sha(data)})
            if member.filename.startswith("Metadaten_") and member.filename.endswith(".txt"):
                dest = DOCS / "original_metadata" / path.stem / Path(member.filename).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    assert dest.read_bytes() == data, "immutable metadata changed"
                else:
                    dest.write_bytes(data)
                result["metadata"].append(dest.relative_to(ROOT).as_posix())
            if not member.filename.startswith("produkt_"):
                continue
            reader = csv.DictReader(io.StringIO(data.decode("latin-1"), newline=None), delimiter=";")
            original_header = list(reader.fieldnames)
            header = [field.strip() for field in original_header]
            qn_field = next(field for field in header if field.startswith("QN_"))
            value_fields = [field for field in header if field not in {"STATIONS_ID", "MESS_DATUM", qn_field, "eor"}]
            qn, qn2024, years, minus999 = Counter(), Counter(), Counter(), Counter()
            rows2024, sample2024, times2024, complete2024 = 0, [], set(), 0
            non_sentinel = Counter()
            for original in reader:
                row = {key.strip(): value.strip() for key, value in original.items()}
                years[row["MESS_DATUM"][:4]] += 1
                qn[row[qn_field]] += 1
                if not row["MESS_DATUM"].startswith("2024"):
                    continue
                rows2024 += 1
                times2024.add(row["MESS_DATUM"])
                qn2024[row[qn_field]] += 1
                for field in value_fields:
                    if float(row[field]) == -999:
                        minus999[field] += 1
                    else:
                        non_sentinel[field] += 1
                required = ("P0",) if "P0" in row else ("TT_TU", "RF_TU")
                complete2024 += all(float(row[field]) != -999 for field in required)
                if len(sample2024) < 2:
                    sample2024.append(row)
            result["products"].append({
                "member": member.filename, "sha256": sha(data), "header_as_published": original_header,
                "header_after_whitespace_trim": header, "rows_by_year": dict(sorted(years.items())),
                "qn_counts_full_product": dict(sorted(qn.items())),
                "rows_2024": rows2024, "unique_timestamp_strings_2024": len(times2024),
                "qn_counts_2024": dict(sorted(qn2024.items())),
                "raw_minus999_counts_2024": dict(minus999),
                "numeric_nonminus999_counts_2024": dict(non_sentinel),
                "rows_2024_with_required_fields_nonminus999": complete2024,
                "qb_columns_present": [field for field in header if field.startswith("QB")],
                "original_value_examples_2024": sample2024,
                "count_caveat": "Counts are raw published values, not accepted independent unedited measurements or fault-free labels.",
            })
    return result


report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "station": {"id": "00433", "published_numeric_id": "433", "name": "Berlin-Tempelhof"},
    "decision": "NOT_YET_ELIGIBLE_FOR_STRICT_ORIGINAL_THREE_CHANNEL_TRAINING",
    "viable_for_strict_real_observation_training_now": False,
    "live_access_verified": True,
    "scope": "Two official DWD product descriptions, CDC terms, station00433 historical TU/P0 ZIPs and contained instrument/parameter metadata; sample counts cover all published2024 records in these two archives.",
    "provider": "Deutscher Wetterdienst (DWD), Climate Data Center",
    "licence": "CC BY 4.0, explicitly stated in both product PDFs and official CDC Terms_of_use.txt/PDF (May2024).",
    "attribution": "Deutscher Wetterdienst (DWD), Climate Data Center; Hourly station observations of 2m air temperature and humidity for Germany / Hourly station observations of pressure for Germany; descriptions v24.03; exact archive URLs, retrieval times and hashes in retrieval_manifest.json. Preserve attribution and identify processing changes in derived outputs.",
    "verified_export_schema": {
        "temperature": {"product": "TU", "raw_field": "TT_TU", "unit": "degC"},
        "humidity": {"product": "TU", "raw_field": "RF_TU", "unit": "percent", "independent_exported_measurement_origin_verified": False},
        "station_pressure": {"product": "P0", "raw_field": "P0", "unit": "hPa"},
        "sea_level_pressure_excluded": {"product": "P0", "raw_field": "P", "unit": "hPa"},
        "join_keys_if_eligible": ["STATIONS_ID", "MESS_DATUM"],
        "timestamp_format": "YYYYMMDDHH; metadata establish UTC for sampled2024 TU and P0; old TU timestamps before1995-02-01 are MEZ and must not be silently treated as UTC.",
        "delimiter": ";", "row_end_marker": "eor", "whitespace": "strip surrounding field whitespace for parsing; preserve original bytes",
        "provider_quality": "QN_9 (TU), QN_8 (P0) are report-level quality-processing stages; no QB columns in either actual product.",
    },
    "verified_station_instrument_history": {
        "temperature": "PT100 electronic air temperature since1994-01-03,2m height; station-height changes separately dated in metadata.",
        "station_pressure": "Digitalbarometer AIR-DB1994-01-03..2015-09-23; PTB330 since2015-09-24.",
        "humidity": "Lithium_Chlorid-Taupunktfuehler(dew-point probe)1994-01-03..2008-11-26; HMP45D2008-11-27..2014-09-17; EE332014-09-18..2019-05-23 and2019-05-28 onward, with later metadata intervals and a short2025 HMP45D overlap.",
    },
    "blocking_findings": [
        "Actual modern TU parameter metadata say RF_TU is generated from SYNOP reports. The inspected materials do not establish whether the exported humidity value is preserved from the humidity sensor or recomputed via dew point; an installed RH probe alone does not prove field-level export lineage.",
        "All 8784 sampled 2024 records in each product have QN3, defined as automatic control and correction. Official docs distinguish QB2 corrected and QB4 added/calculated, but actual files lack QB. The downloaded data therefore cannot identify and exclude edited/calculated individual values under the strict acceptance policy.",
        "No independent hardware-fault or reviewed-normal labels were supplied in these archives. QN codes cannot become binary fault ground truth.",
    ],
    "allowed_next_step": "Resolve exact hourly RF_TU export origin and obtain per-value originality/edit information, or verify a different product that preserves direct channels and flags. Do not implement a training admission adapter that assumes these missing semantics.",
    "samples": [inspect(path) for path in sorted(RAW.glob("*.zip"))],
}
(DOCS / "dwd_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"decision": report["decision"], "products": [{"archive": sample["archive"], "products": [{key: product[key] for key in ("member", "rows_2024", "qn_counts_2024", "raw_minus999_counts_2024", "numeric_nonminus999_counts_2024", "qb_columns_present")} for product in sample["products"]]} for sample in report["samples"]]}, indent=2))
