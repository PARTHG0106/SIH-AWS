"""Build the isolated Indian-station demo from archived native ISD reports.

No legacy preparation, injection, training or real-data artifacts are used.
All synthetic scenarios are generated explicitly by the separate dashboard page.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awsad.demo.indian_stations import DEFAULT_STATION_IDS, build_demo_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/noaa_isd")
    parser.add_argument("--out", type=Path, default=ROOT / "data/indian_demo_20260929")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--stations", nargs="+", default=list(DEFAULT_STATION_IDS),
                        help="11-digit USAF+WBAN keys; each requires an IN metadata row and a native file.")
    args = parser.parse_args()
    manifest = build_demo_bundle(args.raw_dir, args.out, year=args.year,
                                 station_ids=args.stations, project_root=ROOT)
    print(json.dumps({"output_dir": str(args.out.resolve()), "station_count": manifest["station_count"],
                      "native_records": manifest["row_count"], "dataset_policy": manifest["dataset_policy"],
                      "eligible_for_real_training": manifest["eligible_for_real_training"]}, indent=2))


if __name__ == "__main__":
    main()
