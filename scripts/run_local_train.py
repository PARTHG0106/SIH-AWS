"""Local training runner (same code path as the Kaggle notebook).

Examples:
  # fast CPU smoke test on 3 station|source groups (no LSTM):
  python scripts/run_local_train.py --groups 42182099999^noaa_isd 42705699999^openmeteo --no-lstm

  # full local run (uses torch on CUDA if available):
  python scripts/run_local_train.py

By default the headline run uses NOAA ISD groups only. Pass
`--include-reference` to include Open-Meteo groups for an explicitly mixed
development experiment; they are reanalysis references, not AWS sensor truth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awsad.data.training_contract import (  # noqa: E402
    TrainingDataContractError, require_eligible_training_data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="station^source pairs, e.g. 42182099999^noaa_isd")
    ap.add_argument("--max-groups", type=int, default=None,
                    help="randomly sample this many groups (seed 0)")
    ap.add_argument("--include-reference", action="store_true",
                    help="include Open-Meteo reference groups when sampling all groups")
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--lstm-epochs", type=int, default=25)
    ap.add_argument("--lstm-windows", type=int, default=65536)
    args = ap.parse_args()

    try:
        require_eligible_training_data(
            args.processed, report_path=Path(args.out) / "training_preflight.json")
    except TrainingDataContractError as exc:
        print(str(exc), file=sys.stderr)
        print(f"Preflight report: {Path(args.out).resolve() / 'training_preflight.json'}")
        raise SystemExit(2) from None
    from awsad.train_pipeline import load_processed, run_pipeline, split_groups

    dfs = load_processed(args.processed)
    # Keep the default local result aligned with the Kaggle headline protocol:
    # NOAA is the observation proxy; Open-Meteo is reference/reanalysis data.
    all_groups = split_groups(dfs) if (args.groups or args.include_reference) \
        else split_groups(dfs, sources=["noaa_isd"])
    if args.groups:
        want = {g.replace("^", "|") for g in args.groups}
        groups = [g for g in all_groups if g in want]
    elif args.max_groups:
        rng = __import__("numpy").random.default_rng(0)
        groups = sorted(rng.choice(all_groups, size=min(args.max_groups, len(all_groups)),
                                   replace=False).tolist())
    else:
        groups = all_groups
    print(f"Training groups: {len(groups)}")

    run_pipeline(
        dfs, groups, args.out,
        train_deep=not args.no_lstm,
        deep_cfg={
            "lstm_ae_168": {"enabled": False},
            "lstm_ae_24": {"epochs": args.lstm_epochs,
                           "windows_per_epoch": args.lstm_windows},
            "tx_ae": {"epochs": args.lstm_epochs,
                      "windows_per_epoch": max(4096, args.lstm_windows // 2)},
            "forecaster": {"epochs": args.lstm_epochs,
                           "windows_per_epoch": max(4096, args.lstm_windows // 2)},
        },
    )


if __name__ == "__main__":
    main()
