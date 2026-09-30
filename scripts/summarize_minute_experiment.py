"""Summarize frozen minute replay evidence without selecting or retuning models."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from awsad.minute_detection import CHANNELS, SIGNALS, write_json


def summarize(directory):
    root = Path(directory)
    metrics = json.loads((root / "metrics.json").read_text())
    events = pd.read_csv(root / "candidate_events.csv")
    groups = []
    for path in sorted((root / "scored_observations").glob("*.parquet")):
        columns = ["group", "timestamp", "is_candidate", *(c + "__reason_codes" for c in CHANNELS)]
        data = pd.read_parquet(path, columns=columns)
        group = data.group.iloc[0]
        record = next((row for row in groups if row["group"] == group), None)
        if record is None:
            record = {"group": group, "observed_minutes": 0, "candidate_minutes": 0,
                      "candidate_minutes_outside_hourly_selection": 0,
                      "signal_channel_exceedances": dict.fromkeys(SIGNALS, 0)}
            groups.append(record)
        record["observed_minutes"] += len(data)
        record["candidate_minutes"] += int(data.is_candidate.sum())
        record["candidate_minutes_outside_hourly_selection"] += int((data.is_candidate & data.timestamp.dt.minute.ne(0)).sum())
        for c in CHANNELS:
            reason = data[c + "__reason_codes"].str.split("|")
            for signal in SIGNALS:
                record["signal_channel_exceedances"][signal] += int(reason.map(lambda values: signal in values).sum())
    for group in groups:
        group["candidate_percent"] = 100 * group["candidate_minutes"] / group["observed_minutes"]
        group["event_proposals"] = int(events.group.eq(group["group"]).sum())
    result = {"fresh_period": metrics["fresh_period"], "native_observations": metrics["native_observations"],
              "candidate_minutes": metrics["candidate_minutes"], "candidate_event_proposals": metrics["candidate_event_proposals"],
              "groups": groups, "real_fault_accuracy": None, "known_hardware_fault_labels": 0,
              "interpretation": "Operational candidate counts only; no false-alarm, fault-recall or accuracy estimate"}
    write_json(root / "experiment_summary.json", result)
    return result


if __name__ == "__main__":
    for directory in sys.argv[1:]:
        print(json.dumps(summarize(directory), indent=2))
