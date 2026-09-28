"""Download the verified-scope SURFRAD corpus; never run synthetic builders."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awsad.data.fetch_surfrad import fetch_surfrad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "data/raw/surfrad_original")
    parser.add_argument("--stations", nargs="+", default=["bon", "fpk", "gwn"])
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    args = parser.parse_args()
    result = fetch_surfrad(args.archive_dir, stations=tuple(args.stations), years=tuple(args.years))
    print(f"Completed: {len(result['files'])} original daily files, {len(result['failures'])} failures.")
