"""Export two frozen native-minute reference signals for the preliminary edge policy.

This standard-library utility does not fit, tune, calibrate, or evaluate anything.
It copies per-group abrupt-change thresholds and converts elapsed flatline
minutes to seconds, preserving strict-greater comparisons and source provenance.
It does not export the full hub model or establish equivalent detection behavior.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

try:
    from .watchdog import CHANNELS, VERSION, PreliminaryWatchdog
except ImportError:
    from watchdog import CHANNELS, VERSION, PreliminaryWatchdog


def _threshold(record):
    value = record.get("threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("frozen thresholds must be finite, nonnegative numbers")
    if record.get("comparison") != "strictly_greater":
        raise ValueError("edge export requires explicit strictly_greater signal comparisons")
    return float(value)


def export_policy(detector_path, groups=None):
    path = Path(detector_path)
    raw = path.read_bytes()
    detector = json.loads(raw)
    if not detector.get("frozen_at_utc") or detector.get("test_used_for_selection") is not False:
        raise ValueError("export requires a frozen artifact declaring test_used_for_selection=false")
    thresholds = detector["thresholds"]
    selected = list(groups) if groups else sorted(group for group in thresholds if group != "pooled_seen_groups")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("request at least one distinct station|source group")
    result = {}
    for group in selected:
        threshold_group = group if group in thresholds else "pooled_seen_groups"
        if threshold_group not in thresholds:
            raise ValueError("no group-specific or frozen pooled reference exists for " + group)
        steps, flat, evidence = [], [], {}
        for channel in CHANNELS:
            source = thresholds[threshold_group][channel]
            steps.append(_threshold(source["abrupt_change"]))
            flat.append(_threshold(source["flatline_minutes"]) * 60.0)
            evidence[channel] = {"abrupt_change": source["abrupt_change"], "flatline_minutes": source["flatline_minutes"]}
        config = {"cadence_seconds": 60, "grace_seconds": 0, "step_limits": steps,
                  "flatline_seconds": flat, "flatline_epsilon": 0,
                  "policy_origin": "frozen_signal_threshold_export"}
        PreliminaryWatchdog(group, **config)  # validate portable constructor constraints
        result[group] = {"configuration": config, "threshold_group": threshold_group,
                         "uses_pooled_seen_reference": threshold_group != group, "source_threshold_records": evidence}
    cfg = detector.get("config", {})
    return {"schema_version": "edge_frozen_signal_policy_v1", "watchdog_version": VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "groups": result,
            "source": {"detector_path": str(path.resolve()), "detector_sha256": hashlib.sha256(raw).hexdigest(),
                       "frozen_at_utc": detector["frozen_at_utc"], "config_sha256": detector.get("config_sha256"),
                       "models_sha256": detector.get("models_sha256"), "dataset_policy": detector.get("dataset_policy"),
                       "calibration_semantics": detector.get("calibration_semantics"),
                       "train_end": cfg.get("train_end"), "selection_end": cfg.get("selection_end"),
                       "calibration_end": cfg.get("calibration_end"), "test_used_for_selection": False},
            "comparison": "strictly_greater", "channel_order": list(CHANNELS),
            "mapping": {"step_limits": "unchanged absolute adjacent-minute abrupt_change thresholds in channel units",
                        "flatline_seconds": "flatline_minutes threshold multiplied by 60; elapsed time since the first value of an exact-repeat run, with initial duration zero"},
            "semantics": "Provided frozen reference thresholds for two preliminary edge signals, not a separately calibrated edge detector, fault truth or full hub model.",
            "limitations": ["No forecasting, sustained-residual or learned scenario-model signals are exported.",
                            "Range-noticed values reset edge temporal context; hub treatment can differ.",
                            "Reset, gaps and packet timing can change available context.",
                            "Pooled references are explicitly labelled and do not establish calibration for an unseen station.",
                            "No observations are read and no thresholds are fitted or selected by this export."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, help="existing frozen native-minute detector.json")
    parser.add_argument("--groups", nargs="+", help="explicit station|source targets; default existing calibrated groups")
    parser.add_argument("--out", required=True, help="new policy JSON path")
    args = parser.parse_args(argv)
    try:
        policy = export_policy(args.detector, args.groups)
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            json.dump(policy, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"policy": str(target), "groups": list(policy["groups"]), "source_sha256": policy["source"]["detector_sha256"]}, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
