"""Reproduce source-inspection measurements without running downloaded code."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root = Path(__file__).resolve().parent
path = root / "source_receipts.json"
if not path.exists():
    index = json.loads((root / "source_index.json").read_text(encoding="utf-8"))
    for item in index["sources"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
    print(json.dumps({"source_metadata_records": len(index["sources"]),
                     "response_bodies_verified": False,
                     "scope": "Submission contains citation/hash metadata only. Full body audit requires the privately retained receipts or fresh public-source collection."}, indent=2))
    raise SystemExit(0)
doc = json.loads(path.read_text(encoding="utf-8"))
sources = doc["sources"]
verified = 0
for key, item in sources.items():
    if "text" not in item:
        continue
    text = item["text"]
    # Early receipts used requests' ISO-8859-1 default for text/html. Recover
    # the original bytes only when the recorded SHA-256 proves the decoding.
    candidates = ["utf-8", "latin-1", "utf-16", "utf-32"]
    for encoding in candidates:
        try:
            raw = text.encode(encoding)
        except UnicodeEncodeError:
            continue
        if hashlib.sha256(raw).hexdigest() == item["sha256"]:
            try:
                item["text"] = raw.decode("utf-8")
                item["text_encoding"] = "utf-8"
            except UnicodeDecodeError:
                item["text_encoding"] = encoding
            verified += 1
            break
    else:
        raise AssertionError(f"Receipt bytes cannot be reconstructed: {key}")

page = sources["official:sih2026PS"]["text"]
end = page.index("SIH26073")
start = page.rfind("Problem Statement Details", 0, end)
block = re.sub(r"<!--.*?-->", "", page[start:end], flags=re.S)
match = re.search(r">Description</th>\s*<td>(.*?)</td>", block, re.S)
assert match
raw_description = match.group(1)
description = re.sub(r"<br\s*/?>", "\n", raw_description, flags=re.I)
description = html.unescape(re.sub(r"<[^>]+>", "", description))
description = "\n".join(re.sub(r"\s+", " ", line).strip() for line in description.splitlines()).strip()
doc["official_problem_statement"] = {
    "source_receipt": "official:sih2026PS",
    "id": "SIH26073",
    "title": "AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)",
    "organization": "Ministry of Earth Sciences (MoES)",
    "department": "India Meteorological Department",
    "category": "Software",
    "theme": "Disaster Management",
    "description_html": raw_description,
    "description_text": description,
    "extraction": "Uncommented Description cell; HTML tags removed, entities decoded, whitespace normalized. Full original response retained and hash verified.",
}

def parsed(key):
    return json.loads(sources[key]["text"])

aditya = parsed("file:Aditya-Murugan1/SIH26073:artifacts/final_evaluation/metrics.json")
simple = parsed("file:aditya0si/skyguard:evals/results.json")
kestrel = parsed("file:D-Tharun/SIH26073-Team-Kestrel:server/artifacts/evaluation_results.json")
searches = {key[7:]: parsed(key)["total_count"] for key in sources if key.startswith("search:")}
point = aditya["detection"]
recomputed_f1 = 2 * point["tp"] / (2 * point["tp"] + point["fp"] + point["fn"])
assert abs(recomputed_f1 - point["f1"]) < 1e-12
spectral = simple["models"]["Spectral Residual (Fourier FFT)"]
doc["inspection_measurements"] = {
    "method": "Offline parsing of archived public text/JSON; these are source audits and arithmetic checks, not reruns of external models.",
    "verified_response_bodies": verified,
    "github_search_totals": searches,
    "luciefer_generator_rows_from_control_flow": 5 * 60 - 5,
    "luciefer_generator_meteorological_channels": 6,
    "aditya0si_evaluator_reported_channels": 7,
    "aditya0si_spectral_saved_point_f1": spectral["f1"],
    "aditya0si_spectral_saved_point_adjusted_f1": spectral["point_adjusted_f1"],
    "aditya0si_point_adjustment_ratio": spectral["point_adjusted_f1"] / spectral["f1"],
    "aditya_murugan_saved_eval_rows": aditya["eval_rows"],
    "aditya_murugan_recomputed_point_f1": recomputed_f1,
    "aditya_murugan_saved_event_recall": point["event_recall"],
    "kestrel_saved_accuracy_percent": kestrel["accuracy"],
    "kestrel_saved_f1_percent": kestrel["f1_score"],
    "kestrel_saved_precision_percent": kestrel["precision"],
}
path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(doc["inspection_measurements"], indent=2))
print(description)
