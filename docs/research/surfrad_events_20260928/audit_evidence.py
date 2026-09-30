"""Recheck saved provider bytes and summarize samples without creating labels.

Run from any directory: python docs/research/surfrad_events_20260928/audit_evidence.py
The output is research metadata, never training data or confirmed benchmark truth.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def normalized_text(path):
    body = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".html":
        parser = VisibleText()
        parser.feed(body)
        body = " ".join(parser.parts)
    return " ".join(body.split())


def field_summary(rows, value_index, missing_value, flag_index=None):
    values = [float(row[value_index]) for row in rows]
    present = [value for value in values if value != missing_value]
    result = {"present": len(present), "missing": len(values) - len(present),
              "min_present": min(present) if present else None,
              "max_present": max(present) if present else None}
    if flag_index is not None:
        result["raw_qc_counts"] = dict(sorted(Counter(row[flag_index] for row in rows).items()))
    return result


def main():
    sources = []
    source_by_url = {}
    for receipt_path in sorted((HERE / "downloads").glob("*.receipt.json")):
        raw_receipt = receipt_path.read_bytes()
        receipt_hash = hashlib.sha256(raw_receipt).hexdigest()
        if receipt_path.name != receipt_hash + ".receipt.json":
            raise ValueError(f"Receipt bytes changed: {receipt_path}")
        receipt = json.loads(raw_receipt)
        entry = dict(receipt, receipt_path=receipt_path.relative_to(ROOT).as_posix(),
                     receipt_sha256=receipt_hash)
        if "error" not in receipt:
            path = (ROOT / receipt["local_path"]).resolve()
            if not path.is_relative_to(HERE / "downloads"):
                raise ValueError("Evidence path leaves the archived download directory")
            payload = path.read_bytes()
            if len(payload) != receipt["bytes"] or hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
                raise ValueError(f"Provider bytes changed: {path}")
            entry["saved_bytes_verified"] = True
        sources.append(entry)
        source_by_url[receipt["requested_url"]] = entry

    samples = []
    for source in sources:
        url = source["requested_url"]
        if "error" in source:
            continue
        if url.endswith(("gwn97202.dat", "gwn98152.dat")):
            lines = (ROOT / source["local_path"]).read_text().splitlines()
            rows = [line.split() for line in lines[2:] if line.strip()]
            if {len(row) for row in rows} != {48}:
                raise ValueError("Historic SURFRAD sample is not the documented 48-field format")
            times = [datetime(int(r[0]), int(r[2]), int(r[3]), int(r[4]), int(r[5]), tzinfo=timezone.utc) for r in rows]
            samples.append({"url": url, "sha256": source["sha256"], "kind": "historic_surfrad_sample",
                "rows": len(rows), "raw_header": lines[:2], "first_timestamp_utc": times[0].isoformat(),
                "last_timestamp_utc": times[-1].isoformat(),
                "observed_adjacent_differences_seconds": dict(Counter(str(int((b-a).total_seconds())) for a, b in zip(times, times[1:]))),
                "temperature_c": field_summary(rows, 38, -9999.9, 39),
                "relative_humidity_pct": field_summary(rows, 40, -9999.9, 41),
                "pressure_hpa": field_summary(rows, 46, -9999.9, 47),
                "benchmark_accepted": False, "reason": "Historic 3-minute sample outside 2023-2025 scope; evidence investigation only"})
        elif re.search(r"/CRNH0203-2024-(?:IL_Champaign_9_SW|MT_Wolf_Point_29_ENE)\.txt$", url):
            rows = [line.split() for line in (ROOT / source["local_path"]).read_text().splitlines() if line.strip()]
            if {len(row) for row in rows} != {38}:
                raise ValueError("USCRN sample is not the documented 38-field format")
            samples.append({"url": url, "sha256": source["sha256"], "kind": "uscrn_hourly_context_sample",
                "rows": len(rows), "raw_first_date_time": rows[0][1:3], "raw_last_date_time": rows[-1][1:3],
                "wban_ids": sorted({row[0] for row in rows}),
                "temperature_hourly_average_c": field_summary(rows, 9, -9999.0),
                "relative_humidity_hourly_average_pct": field_summary(rows, 26, -9999, 27),
                "pressure_field_available": False, "benchmark_accepted": False,
                "reason": "Provider aggregates for independent context, not same-minute/same-sensor fault truth"})

    quotes_checked = 0
    registry_path = HERE / "event_evidence_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for event in registry["event_evidence"]:
            for quote in event["source_quotes"]:
                source = source_by_url[quote["url"]]
                if quote["sha256"] != source["sha256"]:
                    raise ValueError("Registry references an unverified source hash")
                if " ".join(quote["quote"].split()) not in normalized_text(ROOT / source["local_path"]):
                    raise ValueError(f"Quote not found in saved bytes: {event['evidence_id']}")
                quotes_checked += 1

    output = {"schema_version": 1, "audit_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Saved provider retrievals only; no completeness claim for all maintenance records",
              "successful_retrievals": sum("error" not in r for r in sources),
              "failed_retrievals": sum("error" in r for r in sources),
              "registry_quotes_verified": quotes_checked, "sources": sources, "sample_checks": samples}
    (HERE / "source_manifest.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
