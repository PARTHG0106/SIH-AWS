"""Acquire the prespecified February 2025 test originals without inspecting values."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awsad.data.surfrad_native import fetch_surfrad_date_range


if __name__ == "__main__":
    target = ROOT / "data/raw/surfrad_sih_test_202502"
    result = fetch_surfrad_date_range(target, start="2025-02-01T00:00:00Z",
        end="2025-03-01T00:00:00Z")
    print(f"Archived {len(result['files'])} originals; failures={len(result['failures'])}. "
          "Values have not been inspected for selection.")
