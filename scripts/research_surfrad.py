"""Archive bounded public NOAA SURFRAD evidence without changing observations."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/research/surfrad_20260926"
RAW = ROOT / "data/raw/surfrad_verified_research"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.lines, self.skip = [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.lines.append(data.strip())


def fetch(spec, raw=False):
    name, url = spec.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Use a simple output basename")
    folder = RAW if raw else DOCS
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    at = datetime.now(timezone.utc).isoformat()
    result = {"url": url, "retrieved_at_utc": at, "local_path": str(target.relative_to(ROOT))}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "SkyGuard-source-research/1.0"})
        with urllib.request.urlopen(request, timeout=25) as response:
            result.update({"final_url": response.url, "http_status": response.status,
                           "content_type": response.headers.get("Content-Type"),
                           "last_modified": response.headers.get("Last-Modified")})
            data = response.read(5_000_001)
        if len(data) > 5_000_000:
            raise ValueError("Source exceeds research download limit")
        digest = hashlib.sha256(data).hexdigest()
        if target.exists() and target.read_bytes() != data:
            target = folder / (target.stem + "_" + digest[:12] + target.suffix)
            result["local_path"] = str(target.relative_to(ROOT))
        target.write_bytes(data)
        result.update({"bytes": len(data), "sha256": digest})
        content = data.decode("utf-8", errors="replace")
        if "html" in (result.get("content_type") or ""):
            page = Page()
            page.feed(content)
            result["text_excerpt"] = "\n".join(page.lines)[-22000:]
            result["links"] = sorted(set(page.links))
        else:
            result["text_excerpt"] = "\n".join(content.splitlines()[:10])
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("spec", nargs="+")
    args = parser.parse_args()
    DOCS.mkdir(parents=True, exist_ok=True)
    manifest_path = DOCS / "acquisition_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda item: fetch(item, args.raw), args.spec))
    previous["files"].extend({k: v for k, v in result.items() if k not in {"links", "text_excerpt"}}
                             for result in results)
    manifest_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
