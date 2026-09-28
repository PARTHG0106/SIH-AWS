"""Package an already verified real-observation builder release for Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def package(bundle, destination):
    root, out = Path(bundle).resolve(), Path(destination).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_role") != "processed_training_bundle" or manifest.get("training_data_policy") != "real_observations_only":
        raise ValueError("expected a real-observation builder release")
    preflight = json.loads((root / "data_preflight.json").read_text(encoding="utf-8"))
    if preflight.get("eligible") is not True:
        raise ValueError("the builder's complete source-specific verification must pass first")
    expected = manifest["sha256"]
    actual = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() and p != root / "MANIFEST.json"}
    if set(actual) != set(expected):
        raise ValueError("release files changed after the builder completed")
    for name, path in actual.items():
        if not path.resolve().is_relative_to(root) or digest(path) != expected[name]:
            raise ValueError("release content changed: " + name)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "skyguard-real-observations.zip"
    if archive.exists():
        raise FileExistsError("preserve the previous package; choose a new destination")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        handle.write(root / "MANIFEST.json", "MANIFEST.json")
        for name, path in sorted(actual.items()):
            handle.write(path, name)
    metadata = {"id": "krishnagupta02468/skyguard-sih26073-processed",
        "title": "SkyGuard SIH26073 - Verified SURFRAD observations",
        "licenses": [{"name": "other"}],
        "description": "Original NOAA SURFRAD observations with raw archives, independent measured RH, station pressure and traceable source code. No synthetic observations or fault labels. SURFRAD data CC0; bundled project source retains its own rights."}
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest(archive),
            "build_id": manifest["build_id"], "observed_rows": manifest["observed_rows"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.bundle, args.destination), indent=2))
