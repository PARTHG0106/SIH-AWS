"""Stage the source-only bundle used by the online Kaggle data builder.

Produces:
  data/kaggle_staging/skyguard_source_bundle/  (unzipped layout)
  data/kaggle_staging/skyguard-sih26073.zip    (upload this to Kaggle Datasets)
  data/kaggle_staging/dataset-metadata.json    (optional Kaggle API metadata)

This package excludes all observation files and credentials. The online CPU
builder acquires verified-scope SURFRAD originals and preserves their lineage.
Legacy injected datasets remain ineligible for the real training notebook.

Run: python scripts/prepare_kaggle_upload.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "kaggle_staging"
BUNDLE = STAGE / "skyguard_source_bundle"
LEGACY_STAGED_BUNDLE = STAGE / "awsad_bundle"
EXPECTED_PARSER_VERSION = "surfrad-observed-v1"
SOURCE_ROLES = {
    "noaa_surfrad": "published measured temperature, independent RH and station pressure; provider QC is not hardware-fault truth",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def configured_dataset_id() -> str:
    """Reuse the configured private Kaggle dataset id."""
    published = ROOT / "data" / "kaggle_upload" / "dataset-metadata.json"
    if published.exists():
        dataset_id = json.loads(published.read_text(encoding="utf-8")).get("id")
        if dataset_id:
            return str(dataset_id)
    kernel = ROOT / "kaggle_builder" / "kernel-metadata.json"
    if kernel.exists():
        sources = json.loads(kernel.read_text(encoding="utf-8")).get(
            "dataset_sources", []
        )
        if sources:
            return str(sources[0])
    return "YOUR_KAGGLE_USERNAME/skyguard-sih26073"


def source_parser_version() -> str:
    """Read the parser version without importing pandas or project packages."""
    parser_path = ROOT / "src/awsad/data/surfrad.py"
    tree = ast.parse(parser_path.read_text(encoding="utf-8"), filename=str(parser_path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(t, ast.Name) and t.id == "PARSER_VERSION" for t in targets):
            return str(ast.literal_eval(node.value))
    raise RuntimeError(f"PARSER_VERSION not found in {parser_path}")


def copytree_clean(src: Path, dst: Path, excludes=(".git", "__pycache__")) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if any(part in excludes for part in item.parts):
            continue
        if item.is_dir() or item.suffix in (".bak", ".pyc", ".dat", ".csv", ".parquet", ".zip", ".gz"):
            continue
        out = dst / item.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, out)


def main() -> None:
    parser_version = source_parser_version()
    if parser_version != EXPECTED_PARSER_VERSION:
        raise RuntimeError(
            f"Refusing to package parser {parser_version!r}; "
            f"expected {EXPECTED_PARSER_VERSION!r}"
        )

    required = [
        ROOT / "src/awsad/__init__.py",
        ROOT / "configs",
        ROOT / "docs",
        ROOT / "README.md",
        ROOT / "requirements.txt",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"source bundle inputs are incomplete: {missing}")

    # Remove the obsolete generated staging directory so it cannot be mistaken
    # for the uploadable source bundle. This never touches data/processed.
    stage_root = STAGE.resolve()
    for target in (LEGACY_STAGED_BUNDLE, BUNDLE):
        resolved = target.resolve()
        if resolved.parent != stage_root or not stage_root.is_relative_to(ROOT.resolve()):
            raise RuntimeError(f"Refusing to replace a path outside the staging directory: {resolved}")
        if target.exists():
            shutil.rmtree(target)
    BUNDLE.mkdir(parents=True)

    copytree_clean(ROOT / "src", BUNDLE / "src")
    copytree_clean(ROOT / "configs", BUNDLE / "configs")
    copytree_clean(ROOT / "docs", BUNDLE / "docs")
    (BUNDLE / "scripts").mkdir()
    for name in ("run_real_observation_train.py", "fetch_real_surfrad.py"):
        shutil.copy2(ROOT / "scripts" / name, BUNDLE / "scripts" / name)
    shutil.copy2(ROOT / "README.md", BUNDLE / "README.md")
    shutil.copy2(ROOT / "requirements.txt", BUNDLE / "requirements.txt")

    hashes = {
        str(path.relative_to(BUNDLE)).replace("\\", "/"): sha256_file(path)
        for path in sorted(BUNDLE.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "project": "SkyGuard AI - SIH26073",
        "schema_version": 1,
        "bundle_role": "online_builder_source",
        "processed_data_included": False,
        "training_data_policy": "real_observations_only",
        "migration_status": "verified_scope_surfrad_observation_training",
        "license": "mixed; see docs/LICENSES.md and component licenses",
        "parser_version": parser_version,
        "source_roles": SOURCE_ROLES,
        "contents": [
            "src/awsad source",
            "configs",
            "documentation and provenance",
        ],
        "next_step": (
            "This is the source stage for kaggle/aws_data_builder.ipynb. "
            "Build verified SURFRAD observations, publish the resulting processed "
            "bundle, then run kaggle/aws_anomaly_training.ipynb."
        ),
        "sources": {
            "noaa_surfrad": "https://gml.noaa.gov/aftp/data/radiation/surfrad/",
        },
        "sha256": hashes,
        "built_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }
    (BUNDLE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    zip_path = shutil.make_archive(
        str(STAGE / "skyguard-sih26073"), "zip", BUNDLE
    )
    metadata = {
        "title": "SkyGuard AI - SIH26073 builder source",
        "id": configured_dataset_id(),
        "licenses": [{"name": "other"}],
        "description": (
            "Source-only bootstrap for the online SkyGuard AI data builder. "
            "Builds original NOAA SURFRAD measurements with immutable raw files, "
            "field lineage and provider QC. No synthetic observations or fault labels."
        ),
    }
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    size_mb = Path(zip_path).stat().st_size / 1e6
    file_count = sum(1 for path in BUNDLE.rglob("*") if path.is_file())
    print(f"\nBundle: {BUNDLE}")
    print(f"Zip:    {zip_path}  ({size_mb:.1f} MB)")
    print(f"Files:  {file_count}")
    print("Role:   online_builder_source (not valid as GPU training input)")
    print(
        "\nSource package refreshed locally. The source -> builder -> processed -> "
        "training chain is preserved. Nothing was uploaded or launched."
    )


if __name__ == "__main__":
    main()
