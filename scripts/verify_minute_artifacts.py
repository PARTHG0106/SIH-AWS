"""Verify artifact hashes and exact equality of every archived input column."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from awsad.data.acquisition import sha256_file
from awsad.minute_detection import write_json


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact manifest path escapes its directory")
    return path


def verify(artifact_dir, native_cache):
    art, cache = Path(artifact_dir).resolve(), Path(native_cache).resolve()
    detector = json.loads((art / "detector.json").read_text())
    provenance = json.loads((art / "provenance.json").read_text())
    metrics = json.loads((art / "metrics.json").read_text())
    native = json.loads((cache / "native_manifest.json").read_text())
    if native != provenance["native_manifest"]:
        raise ValueError("native input manifest differs from the replay's recorded source")
    if sha256_file(art / "models.joblib") != detector["models_sha256"]:
        raise ValueError("model bytes no longer match the frozen detector")
    for name, digest in detector["source_files"].items():
        if sha256_file(inside(art / "source_snapshot", name)) != digest:
            raise ValueError("frozen source snapshot differs")
    raw_index = {entry["file"]: entry for entry in native["shards"]}
    rows, labels, columns = 0, 0, set()
    for scored in provenance["scored_shards"]:
        path = inside(art, scored["file"])
        original = raw_index[path.name]
        native_path = inside(cache, original["file"])
        if sha256_file(path) != scored["sha256"] or sha256_file(native_path) != original["sha256"]:
            raise ValueError("native/scored artifact bytes differ from the recorded manifest")
        before = pd.read_parquet(native_path)
        after = pd.read_parquet(path, columns=list(before))
        pd.testing.assert_frame_equal(before, after, check_exact=True)
        columns.update(before.columns)
        rows += len(before)
        labels += int(before.label.notna().sum())
    if rows != metrics["native_observations"] or labels:
        raise ValueError("artifact row/unknown-label accounting failed")
    report = {"status": "verified", "native_rows_compared_exactly": rows,
              "original_columns_preserved": len(columns), "known_hardware_fault_labels": labels,
              "scored_shards_verified": len(provenance["scored_shards"]),
              "frozen_source_files_verified": len(detector["source_files"]),
              "models_sha256": detector["models_sha256"],
              "metrics_sha256": sha256_file(art / "metrics.json"),
              "provenance_sha256": sha256_file(art / "provenance.json"),
              "scope": "Exact input/output equality against cache reconstructed from immutable original daily files"}
    write_json(art / "artifact_verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("native_cache", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact_dir, args.native_cache), indent=2))
