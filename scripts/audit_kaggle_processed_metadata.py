"""Read-only inventory and bounded metadata downloads for a Kaggle dataset.

Never downloads observation files, starts kernels, publishes datasets, or
prints credentials/API error bodies. The raw downloaded JSON is evidence,
not an acceptance of its claims about observation or label provenance.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggle_manage import ROOT, api_client

ALLOWED_NAMES = {"MANIFEST.json", "source_metadata.json", "splits.json"}
MAX_BYTES = 2_000_000


def safe_error(exc):
    response = getattr(exc, "response", None)
    return {"type": type(exc).__name__, "http_status": getattr(response, "status_code", None)}


def inventory(api, dataset):
    status = json.loads(api.dataset_status(dataset, format="json"))
    version = status.get("current_version_number")
    pinned = f"{dataset}/{version}" if version else dataset
    files, token = [], None
    for _ in range(100):
        response = api.dataset_list_files(pinned, page_token=token, page_size=100)
        for item in response.files or []:
            row = {"name": str(getattr(item, "name", ""))}
            for key in ("total_bytes", "size", "creation_date", "last_updated"):
                value = getattr(item, key, None)
                if value is not None:
                    row[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
            files.append(row)
        token = getattr(response, "next_page_token", None)
        if not token:
            break
    return {"dataset": dataset, "pinned_dataset": pinned, "status": status,
            "files": files, "inventory_complete": not bool(token)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, default=ROOT / "kaggle-krishna.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research/kaggle_processed_20260925")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "read_only": True, "download_max_bytes": MAX_BYTES,
              "allowed_download_names": sorted(ALLOWED_NAMES), "datasets": [], "downloads": []}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            api = api_client(args.credentials)
        for dataset in ("krishnagupta02468/skyguard-sih26073", "krishnagupta02468/skyguard-sih26073-processed"):
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    item = inventory(api, dataset)
                result["datasets"].append(item)
            except Exception as exc:
                result["datasets"].append({"dataset": dataset, "error": safe_error(exc)})
        if args.download:
            for dataset in result["datasets"]:
                for row in dataset.get("files", []):
                    name = Path(row["name"]).name
                    if name not in ALLOWED_NAMES:
                        continue
                    size = row.get("total_bytes", row.get("size"))
                    entry = {"remote_file": row["name"], "listed_bytes": size,
                             "dataset": dataset["pinned_dataset"]}
                    if not isinstance(size, (float, int)) or not 0 <= size <= MAX_BYTES:
                        entry["skipped"] = "unknown or excessive remote size"
                        result["downloads"].append(entry)
                        continue
                    folder = args.output / f"download_{len(result['downloads']) + 1}"
                    folder.mkdir(exist_ok=True)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            api.dataset_download_file(dataset["pinned_dataset"], row["name"],
                                                      path=str(folder), force=False, quiet=True)
                        downloaded = list(folder.iterdir())
                        if len(downloaded) != 1 or downloaded[0].name != name:
                            entry["error"] = {"type": "UnexpectedMetadataDownloadName"}
                        else:
                            file = downloaded[0]
                            payload = file.read_bytes()
                            entry.update({"local_path": str(file.relative_to(args.output)),
                                          "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
                            if len(payload) > MAX_BYTES:
                                entry["error"] = {"type": "MetadataSizeExceeded"}
                            else:
                                parsed = json.loads(payload)
                                if isinstance(parsed, dict):
                                    entry["top_level_keys"] = sorted(parsed)
                                    for key in ("build_id", "built_at", "created_at", "generated_at", "bundle_type",
                                                "dataset_version", "source_manifest_sha256", "parser_version"):
                                        if key in parsed:
                                            entry[key] = parsed[key]
                    except Exception as exc:
                        entry["error"] = safe_error(exc)
                    result["downloads"].append(entry)
    except Exception as exc:
        result["error"] = safe_error(exc)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "datasets"}
    summary["datasets"] = [
        {**{key: value for key, value in row.items() if key != "files"},
         "file_count": len(row.get("files", [])),
         "metadata_files": [item for item in row.get("files", [])
                            if Path(item["name"]).name in ALLOWED_NAMES]}
        for row in result["datasets"]]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
