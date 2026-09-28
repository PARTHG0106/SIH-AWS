"""Paired uncertainty estimates from existing held-out forecast errors only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(directory: Path) -> dict:
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if not report["config"]["evaluate_test"]:
        raise ValueError("only summarize the frozen final test run")
    frame = pd.read_parquet(directory / "scored_observations.parquet")
    lookup = frame.set_index("observation_id")
    result = {"experiment": report["experiment"], "comparison": "validation-selected predictor versus observed persistence on identical accepted test targets",
              "confidence_interval": "paired moving-block bootstrap of observed daily error totals, 7-day blocks, 2000 resamples, seed42; conditional on this corpus/model",
              "synthetic_training_observations": 0, "results": []}
    for horizon, data in report["horizons"].items():
        origin_ids = frame[f"{horizon}__context_id__origin_minus_0m"]
        for channel, model in data["channels"].items():
            if model.get("status") != "trained":
                continue
            y = frame[channel].to_numpy(float, na_value=np.nan)
            prediction = frame[f"{horizon}__{channel}__prediction"].to_numpy(float)
            baseline = origin_ids.map(lookup[channel]).to_numpy(float, na_value=np.nan)
            for partition in ("seen", "holdout"):
                valid = (frame.evaluation_split.eq("test") & frame.station_partition.eq(partition)
                         & frame[f"{horizon}__features_available"] & frame[f"{horizon}__context_provider_accepted"]
                         & frame[channel + "__qc_accepted"].fillna(False)
                         & np.isfinite(y) & np.isfinite(prediction) & np.isfinite(baseline)).to_numpy(bool)
                if not valid.any():
                    continue
                errors = pd.DataFrame({"date": frame.loc[valid, "timestamp"].dt.floor("D"),
                    "selected": np.abs(y[valid] - prediction[valid]),
                    "persistence": np.abs(y[valid] - baseline[valid]), "n": 1})
                daily = errors.groupby("date")[["selected", "persistence", "n"]].sum().to_numpy(float)
                n_days = len(daily)
                rng = np.random.default_rng(42)
                width = min(7, n_days)
                bootstrap = []
                for _ in range(2000):
                    starts = rng.integers(0, n_days - width + 1, size=int(np.ceil(n_days / width)))
                    indices = (starts[:, None] + np.arange(width)[None, :]).ravel()[:n_days]
                    selected_sum, baseline_sum, count = daily[indices].sum(axis=0)
                    bootstrap.append([(baseline_sum - selected_sum) / count,
                                      100 * (baseline_sum - selected_sum) / baseline_sum if baseline_sum else 0])
                intervals = np.quantile(np.asarray(bootstrap), [.025, .975], axis=0)
                selected_mae, baseline_mae = errors.selected.mean(), errors.persistence.mean()
                result["results"].append({"horizon": horizon, "channel": channel, "station_partition": partition,
                    "selected_model": model["selected_model"], "n": int(valid.sum()), "days": n_days,
                    "selected_mae": float(selected_mae), "persistence_mae": float(baseline_mae),
                    "mae_improvement_pct": float(100 * (baseline_mae - selected_mae) / baseline_mae) if baseline_mae else None,
                    "mae_reduction_95pct_interval": intervals[:, 0].tolist(),
                    "improvement_pct_95pct_interval": intervals[:, 1].tolist()})
    output = directory / "paired_test_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
