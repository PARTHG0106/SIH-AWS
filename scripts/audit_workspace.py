"""Read-only inventory of data and the active Python environment."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "torch",
                 "pytest", "streamlit", "nbformat", "kaggle"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    result["raw"] = {}
    for folder in sorted((ROOT / "data/raw").iterdir()):
        if not folder.is_dir():
            continue
        files = [p for p in folder.rglob("*") if p.is_file() and ".git" not in p.parts]
        result["raw"][folder.name] = {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    try:
        import pyarrow.parquet as pq
        result["processed"] = {}
        for path in sorted((ROOT / "data/processed").glob("*.parquet")):
            pf = pq.ParquetFile(path)
            result["processed"][path.name] = {"rows": pf.metadata.num_rows,
                "row_groups": pf.num_row_groups, "bytes": path.stat().st_size,
                "columns": pf.schema_arrow.names}
    except ImportError:
        pass
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
