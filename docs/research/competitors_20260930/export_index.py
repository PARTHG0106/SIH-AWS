"""Export citation/hash metadata without redistributing inspected source bodies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / "source_receipts.json"
receipts = json.loads(path.read_text(encoding="utf-8"))
fields = ("requested_url", "final_url", "retrieved_at_utc", "status", "sha256",
          "bytes", "content_type", "text_encoding")
index = {"research_date": receipts["research_date"],
         "source_receipts_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "scope": "Citation and response-hash metadata only. Inspected third-party source bodies are not redistributed in the submission package.",
         "sources": {key: {field: value[field] for field in fields if field in value}
                     for key, value in receipts["sources"].items()}}
(ROOT / "source_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
print(f"Exported {len(index['sources'])} source metadata records.")
