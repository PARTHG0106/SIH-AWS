"""Runtime regressions use synthetic histories and temporary artifact fixtures."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from awsad.benchmark.operational_model import CHANNELS, OperationalPatternModel, causal_pattern_features
from awsad.live_detector import LiveMinuteDetector
from awsad.minute_detection import MinuteConfig, SIGNALS


class _FixtureProbabilities:
    classes_ = np.array(["no_injection", "spike", "drift"])

    def predict_proba(self, features):
        return np.tile([.4, .3, .3], (len(features), 1))


def _history():
    index = np.arange(181, dtype=float)
    return pd.DataFrame({"timestamp": pd.date_range("2038-01-01", periods=181, freq="min", tz="UTC"),
        "station_id": "TEST_ONLY", "source": "software_fixture", "observation_id": [str(i) for i in index],
        **{channel: index + offset for channel, offset in zip(CHANNELS, (20, 900, 50))}})


def _model():
    return OperationalPatternModel(_FixtureProbabilities(), list(causal_pattern_features(_history())), threshold=.5)


@pytest.mark.parametrize("channels", [[CHANNELS[0]], list(CHANNELS)])
def test_json_null_histories_match_numeric_missing_feature_semantics(channels):
    nulls, nans = _history(), _history()
    for channel in channels:
        nulls[channel] = None
        nans[channel] = np.nan
    expected = causal_pattern_features(nans)
    actual = causal_pattern_features(nulls)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_live_pattern_hook_scores_null_history_after_contiguous_warmup():
    cfg = MinuteConfig()
    models = {f"{h}:{c}": None for h in cfg.horizons for c in CHANNELS}
    thresholds = {"pooled_seen_groups": {c: {s: {"threshold": 1000.} for s in SIGNALS} for c in CHANNELS}}
    engine = LiveMinuteDetector(models, thresholds, cfg, pattern_predictor=_model().predict_one)
    history = _history()
    for channel in CHANNELS:
        history[channel] = None
    results = engine.ingest_many(history.to_dict("records"))
    assert results[-2]["pattern_evidence"]["status"] == "warming_up"
    assert results[-1]["pattern_evidence"]["status"] == "scored"
    assert results[-1]["availability"]["missing_channels"] == list(CHANNELS)
    assert results[-1]["hardware_fault_status"] == "unknown"


def test_gap_warmup_and_alert_pattern_confidence_stay_consistent():
    history, model = _history(), _model()
    scored = model.predict_one(history)
    assert scored["is_candidate"]
    assert scored["suggested_pattern"] in ("spike", "drift")
    assert scored["pattern_confidence"] == pytest.approx(.3)
    assert scored["scenario_score"] == pytest.approx(.6)
    history.loc[180, "timestamp"] += pd.Timedelta(minutes=1)
    gap = model.predict_one(history)
    assert gap["status"] == "warming_up"
    assert gap["scenario_score"] is None and not gap["is_candidate"]


def test_api_rejects_valid_model_bytes_with_stale_feature_source_seal(tmp_path, monkeypatch):
    from app import api
    model_path = tmp_path / "pattern_model.joblib"
    model_path.write_bytes(b"not-a-model; loader must never be reached for a stale source seal")
    frozen = {"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "policy": "synthetic_scenarios_on_original_observations",
              "source_fingerprints": {"src/awsad/benchmark/operational_model.py": "0" * 64}}
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(api, "SIH_BENCH", tmp_path)
    monkeypatch.setattr(api, "_pattern_cache", {})
    def unexpected_load(path):
        raise AssertionError("model deserialization was reached despite a stale feature source seal")
    monkeypatch.setattr(api.OperationalPatternModel, "load", unexpected_load)
    with pytest.raises(ValueError, match="source|fingerprint|seal"):
        api._predict_pattern(_history(), {})
