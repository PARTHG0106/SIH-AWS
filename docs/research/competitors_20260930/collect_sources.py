"""Read-only public-source receipts; never imports or runs downloaded code.

Examples (run from repository root):
  .venv/Scripts/python.exe docs/research/competitors_20260930/collect_sources.py search
  .venv/Scripts/python.exe docs/research/competitors_20260930/collect_sources.py trees owner/repo
  .venv/Scripts/python.exe docs/research/competitors_20260930/collect_sources.py files owner/repo README.md file.py
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_receipts.json"
RECEIPTS = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
    "research_date": "2026-09-30",
    "method": "Read-only HTTP; source text inspected, never executed. Hashes cover response bytes.",
    "sources": {},
}


def fetch(key: str, url: str):
    try:
        response = requests.get(url, headers={"User-Agent": "SkyGuard-source-inspection/1.0", "Accept": "application/vnd.github+json" if "api.github.com" in url else "*/*"}, timeout=40)
        body = response.content
        try:
            decoded = body.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            decoded = response.text
            encoding = response.encoding
        receipt = {
            "requested_url": url,
            "final_url": response.url,
            "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": response.status_code,
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "content_type": response.headers.get("Content-Type"),
            "text_encoding": encoding,
            "text": decoded,
        }
        return key, receipt
    except Exception as exc:
        return key, {"requested_url": url, "error": str(exc), "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def batch(tasks):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for key, receipt in pool.map(lambda item: fetch(*item), tasks):
            RECEIPTS["sources"][key] = receipt
            print(key, receipt.get("status"), receipt.get("bytes"), receipt.get("error", ""))
    OUT.write_text(json.dumps(RECEIPTS, indent=2, ensure_ascii=False), encoding="utf-8")


def data(key):
    return json.loads(RECEIPTS["sources"][key]["text"])


command, *args = sys.argv[1:]
if command == "search":
    queries = ["SIH26073", '"automatic weather" anomaly', '"SkyGuard" weather']
    batch([(f"search:{query}", "https://api.github.com/search/repositories?q=" + quote(query) + "&per_page=100") for query in queries] + [("official:sih2026PS", "https://www.sih.gov.in/sih2026PS")])
    for query in queries:
        result = data(f"search:{query}")
        print("QUERY", query, "total", result.get("total_count"))
        for item in result.get("items", []):
            print(item["full_name"], "size", item["size"], "license", (item.get("license") or {}).get("spdx_id"), item.get("description"))
elif command == "trees":
    batch([(f"repo:{repo}", "https://api.github.com/repos/" + repo) for repo in args])
    batch([(f"head:{repo}", "https://api.github.com/repos/" + repo + "/commits/" + data("repo:" + repo).get("default_branch", "main")) for repo in args])
    batch([(f"tree:{repo}", "https://api.github.com/repos/" + repo + "/git/trees/" + data("head:" + repo)["sha"] + "?recursive=1") for repo in args if "sha" in data("head:" + repo)])
    for repo in args:
        print("REPOSITORY", repo, data("head:" + repo).get("sha"))
        key = "tree:" + repo
        if key in RECEIPTS["sources"]:
            for item in data(key).get("tree", []):
                if item["type"] == "blob" and item["path"].lower().endswith((".py", ".ipynb", ".md", ".json", ".cpp", ".h", ".txt", ".yml", ".yaml", "license")):
                    print(item["path"], item.get("size"))
elif command in {"files", "fetch-files"}:
    repo, *paths = args
    commit = data("head:" + repo)["sha"]
    batch([(f"file:{repo}:{path}", f"https://raw.githubusercontent.com/{repo}/{commit}/{quote(path)}") for path in paths])
    if command == "fetch-files":
        raise SystemExit(0)
    for path in paths:
        receipt = RECEIPTS["sources"][f"file:{repo}:{path}"]
        print("FILE", repo, path, receipt.get("status"))
        if path.endswith(".ipynb"):
            notebook = json.loads(receipt["text"])
            for i, cell in enumerate(notebook.get("cells", [])):
                print(f"CELL {i} {cell.get('cell_type')}")
                print("".join(cell.get("source", [])))
        else:
            for i, line in enumerate(receipt.get("text", "").splitlines(), 1):
                print(f"{i}: {line}")
elif command == "fetch":
    key, url = args
    batch([(key, url)])
elif command == "show":
    for key in args:
        bounds = None
        if "@" in key:
            key, bounds = key.rsplit("@", 1)
            bounds = [int(n) for n in bounds.split(":")]
        text = RECEIPTS["sources"][key].get("text", "")
        print("SOURCE", key)
        for i, line in enumerate(text.splitlines(), 1):
            if bounds is None or bounds[0] <= i <= bounds[1]:
                print(f"{i}: {line}")
elif command == "scan":
    import re
    pattern, *keys = args
    for key in keys:
        print("SOURCE", key)
        lines = RECEIPTS["sources"][key].get("text", "").splitlines()
        selected = set()
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.I):
                selected.update(range(max(0, i - 2), min(len(lines), i + 4)))
        for i in sorted(selected):
            print(f"{i + 1}: {lines[i]}")
else:
    raise SystemExit("Expected search, trees, files, fetch or show")
