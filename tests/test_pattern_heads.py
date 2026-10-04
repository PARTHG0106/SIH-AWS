"""Synthetic-only classifier interface and warm-up tests."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from awsad.benchmark.pattern_heads import TwoHeadPatternClassifier
from awsad.benchmark.operational_model import OperationalPatternModel, causal_pattern_features


def test_two_head_joint_probabilities_are_normalized_and_have_background():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 4))
    y = np.array(["no_injection"] * 60 + ["spike"] * 30 + ["drift"] * 30)
    with threadpool_limits(limits=1):
        model = TwoHeadPatternClassifier(max_iter=3, max_leaf_nodes=3).fit(X, y)
        probabilities = model.predict_proba(X)
    assert model.classes_.tolist() == ["no_injection", "drift", "spike"]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
    assert (probabilities >= 0).all()


class FixedModel:
    classes_ = np.array(["no_injection", "spike", "drift"])

    def predict_proba(self, X):
        return np.tile([.4, .3, .3], (len(X), 1))


def frame(n=181):
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
        "temperature_c": np.arange(n) / 10, "pressure_hpa": 1000., "relative_humidity_pct": 50.})


def test_live_pattern_needs_contiguous_context_and_does_not_name_background_as_fault():
    records = frame()
    model = OperationalPatternModel(FixedModel(), list(causal_pattern_features(records)))
    assert model.predict_one(records.iloc[:1])["status"] == "warming_up"
    missing = records.copy()
    missing.loc[50:, "timestamp"] += pd.Timedelta(minutes=1)
    assert model.predict_one(missing)["status"] == "warming_up"
    evidence = model.predict_one(records)
    assert evidence["is_candidate"]
    assert evidence["suggested_pattern"] == "spike"
    np.testing.assert_allclose(evidence["pattern_confidence"], .3)


def test_json_null_history_has_same_features_as_numeric_missing_history():
    records = frame()
    records["temperature_c"] = None
    numeric = records.copy()
    numeric["temperature_c"] = np.nan
    pd.testing.assert_frame_equal(causal_pattern_features(records), causal_pattern_features(numeric))
    model = OperationalPatternModel(FixedModel(), list(causal_pattern_features(numeric)))
    assert model.predict_one(records)["status"] == "scored"


class OrderedModel:
    classes_ = np.array(["no_injection", "spike", "drift"])

    def predict_proba(self, X):
        p = X[:, 0]
        return np.column_stack((1-p, p*.65, p*.35))


def test_separate_probability_calibration_preserves_detection_ranking():
    features = pd.DataFrame({"probability": np.linspace(.01, .99, 90)})
    labels = np.array(["no_injection"] * 50 + ["spike"] * 25 + ["drift"] * 15)
    model = OperationalPatternModel(OrderedModel(), ["probability"])
    model.calibrate(features, labels)
    result = model.predict_features(features)
    assert model.binary_calibration[0] > 0
    assert (np.diff(result["scenario_score"]) >= 0).all()
    np.testing.assert_allclose(result["probabilities"].sum(axis=1), 1.)
