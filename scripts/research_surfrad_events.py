"""Archive primary-source SURFRAD event evidence without creating any labels.

Usage: python scripts/research_surfrad_events.py URL [URL ...]
Each response is content addressed, and every acquisition has its own receipt.
This research utility never writes observations or benchmark truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("docs/research/surfrad_events_20260928"))
    args = parser.parse_args()
    downloads = args.out / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for url in args.urls:
        receipt = {"requested_url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            request = Request(url, headers={"User-Agent": "SkyGuard research/1.0 (primary-source verification)"})
            with urlopen(request, timeout=40) as response:
                payload = response.read()
                digest = hashlib.sha256(payload).hexdigest()
                suffix = ".html" if "html" in response.headers.get("Content-Type", "") else ".txt"
                path = downloads / (digest + suffix)
                if path.exists() and path.read_bytes() != payload:
                    raise ValueError("Content-addressed evidence collision")
                if not path.exists():
                    path.write_bytes(payload)
                receipt.update(final_url=response.url, status=response.status,
                               content_type=response.headers.get("Content-Type"),
                               last_modified=response.headers.get("Last-Modified"),
                               sha256=digest, bytes=len(payload), local_path=path.as_posix())
        except Exception as exc:
            receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        body = json.dumps(receipt, indent=2).encode("utf-8")
        receipt_name = hashlib.sha256(body).hexdigest() + ".receipt.json"
        (downloads / receipt_name).write_bytes(body)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
