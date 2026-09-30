"""Chunking and held-out-station invariants with software-only test fixtures."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from awsad.data.acquisition import sha256_file
from awsad.minute_detection import CHANNELS, SIGNALS, MinuteConfig, compute_signals

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("minute_runner", ROOT / "scripts/run_minute_detection.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def fixture_frame(n=850, station="bon", start="2024-07-01"):
    x = np.arange(n)
    frame = pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="min", tz="UTC"),
        "source": "software_test", "station_id": station,
        "observation_id": [station + "-" + start + "-fixture-" + str(i) for i in x]})
    for i, c in enumerate(CHANNELS):
        frame[c] = np.floor(x / (20 + i * 70))
        frame[c + "__qc_accepted"] = pd.array([True] * n, dtype="boolean")
    frame.loc[200, CHANNELS[2] + "__qc_accepted"] = False
    return frame


def cache_frames(path, frames):
    shards = []
    for i, frame in enumerate(frames):
        filename = f"shard_{i}.parquet"
        frame.to_parquet(path / filename, index=False)
        shards.append({"file": filename, "sha256": sha256_file(path / filename), "rows": len(frame),
            "station_id": frame.station_id.iloc[0], "source": "software_test",
            "start": frame.timestamp.iloc[0].isoformat(), "end": frame.timestamp.iloc[-1].isoformat()})
    return {"shards": shards}


def test_scores_and_quality_are_invariant_to_chunk_boundaries_with_long_window(tmp_path):
    frame = fixture_frame()
    manifest = cache_frames(tmp_path, [frame.iloc[:350], frame.iloc[350:]])
    cfg = replace(MinuteConfig(), sustained_minutes=90)
    models = {f"{h}:{c}": None for h in cfg.horizons for c in CHANNELS}
    whole, whole_qc, whole_predictions, _ = compute_signals(frame, models, cfg)
    chunks = list(RUNNER.signal_walk(tmp_path, manifest, models, cfg))
    for c in CHANNELS:
        for s in SIGNALS:
            np.testing.assert_allclose(np.concatenate([chunk[2][c][s] for chunk in chunks]), whole[c][s], equal_nan=True)
            np.testing.assert_array_equal(np.concatenate([chunk[3][c][s] for chunk in chunks]), whole_qc[c][s])
    for h in cfg.horizons:
        np.testing.assert_allclose(np.concatenate([chunk[4][h] for chunk in chunks]), whole_predictions[h], equal_nan=True)


def test_heldout_station_cannot_change_fit_or_selection_samples(tmp_path):
    train = fixture_frame(start="2024-06-01")
    val = fixture_frame(start="2024-07-01")
    holdout = fixture_frame(station="gwn", start="2024-06-01")
    manifest = cache_frames(tmp_path, [train, val, holdout])
    cfg = MinuteConfig()
    before = RUNNER.collect_samples(tmp_path, manifest, cfg)
    holdout.loc[:, list(CHANNELS)] = 100000.
    manifest = cache_frames(tmp_path, [train, val, holdout])
    after = RUNNER.collect_samples(tmp_path, manifest, cfg)
    for split in before:
        for h in cfg.horizons:
            for key in before[split][h]:
                np.testing.assert_array_equal(before[split][h][key], after[split][h][key])
