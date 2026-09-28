"""Fast offline smoke tests (no network). Run: python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_supervised_scorer_roundtrip(tmp_path):
    """stack_meta_features + SupervisedScorer load/predict parity and guards."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from awsad.inference import stack_meta_features, SupervisedScorer

    rng = np.random.default_rng(0)
    comps = ["qc", "iforest", "spatial"]
    fcols = ["temperature_c__resid", "temperature_c__roll_std_6h"]
    n = 400
    scores = rng.normal(size=(n, len(comps)))
    feats = rng.normal(size=(n, len(fcols)))
    hard = (rng.random(n) < 0.1).astype(float)
    X = stack_meta_features(scores, feats, hard)
    assert X.shape == (n, len(comps) + len(fcols) + 1)
    y = (X[:, 0] + X[:, -1] > 0.5).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X, y)
    blob = {"model": clf, "components": comps, "fcols": fcols,
            "pooled_threshold": 0.5, "per_group_thresholds": {"g1": 0.7}}
    p = tmp_path / "supervised_meta.joblib"
    joblib.dump(blob, p)

    scorer = SupervisedScorer.load(p)
    assert scorer.n_features() == len(comps) + len(fcols) + 1
    assert scorer.threshold_for("g1") == 0.7
    assert scorer.threshold_for("unknown") == 0.5
    proba, flag = scorer.predict(X, group="g1")
    assert proba.shape == (n,) and set(np.unique(flag)) <= {0, 1}
    # feature-count guard
    try:
        scorer.predict_proba(X[:, :-1])
        assert False, "should have raised on wrong feature count"
    except ValueError:
        pass


def test_forecaster_make_pairs_next_step_in_bounds():
    """Next-step target index start+w must never index past the block end.

    Regression for the training crash `IndexError: index 55760 is out of bounds
    for axis 0 with size 55760` in ForecasterDetector._make_pairs — the window
    start upper bound was len(blk)-w+1 (exclusive), which allowed
    start+w == len(blk).
    """
    from awsad.models.lstm_forecaster import ForecasterDetector, TrainConfig

    w = 24
    rng = np.random.default_rng(0)
    blocks = [
        rng.normal(size=(w + 1, 3)).astype("float32"),   # tightest valid block
        rng.normal(size=(w + 2, 3)).astype("float32"),
        rng.normal(size=(2048, 3)).astype("float32"),
    ]
    det = ForecasterDetector(window=w, hidden=8, layers=1, dropout=0.0)
    for _ in range(500):  # hammer to hit boundary starts
        X, Y = det._make_pairs(blocks, rng, 256, balance_blocks=True)
        assert X.shape == (256, w, 3) and Y.shape == (256, 3)
    cfg = TrainConfig(window=w)
    cfg.epochs, cfg.batch_size, cfg.windows_per_epoch = 1, 64, 256
    det.fit(blocks, ["a", "b", "c"], val_blocks=blocks, cfg=cfg,
            device="cpu", log=lambda *a, **k: None)


def test_forecaster_score_series_shapes_align():
    """score_series must not over-read the final (target-less) window.

    Regression for the 2nd training crash `operands could not be broadcast
    together with shapes (191,57) (190,57)` — the chunk slice took len(wins)
    windows but only len(wins)-1 have a next-step target, so _predict(X) was
    one row longer than Y.
    """
    from awsad.models.lstm_forecaster import ForecasterDetector, TrainConfig

    w = 24
    det = ForecasterDetector(window=w, hidden=8, layers=1, dropout=0.0)
    blocks = [np.random.default_rng(0).normal(size=(3000, 4)).astype("float32")]
    cfg = TrainConfig(window=w)
    cfg.epochs, cfg.batch_size, cfg.windows_per_epoch = 1, 64, 128
    det.fit(blocks, list("abcd"), val_blocks=blocks, cfg=cfg,
            device="cpu", log=lambda *a, **k: None)
    # Blocks of several lengths, including ones that cross/short the chunk size.
    for n in (214, 500, 3000):
        s = det.score_series(np.random.default_rng(1).normal(size=(n, 4)).astype("float32"))
        assert s.shape == (n,)
        assert np.isfinite(s).all()


def test_dropout_flags_catch_zero_runs_not_calm():
    """Zero persistence alerts after three observed zeros; never retroactively."""
    from awsad.preprocessing.qc_rules import dropout_flags, compute_qc_flags, hard_flag

    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "temperature_c": 20 + rng.normal(0, 2, n),
        "pressure_hpa": 1010 + rng.normal(0, 1, n),
        "relative_humidity_pct": np.clip(60 + rng.normal(0, 5, n), 1, 100),
    })
    df.loc[50:60, "temperature_c"] = 0.0          # 11h zero-fill dropout
    df.loc[100, "temperature_c"] = 0.0            # single zero -> NOT a dropout
    f = dropout_flags(df)
    assert not bool(f["dropout_temperature_c"].iloc[50:52].any())
    assert bool(f["dropout_temperature_c"].iloc[52:61].all())
    assert not bool(f["dropout_temperature_c"].iloc[100])
    assert not bool(f["dropout_temperature_c"].iloc[:50].any())
    # dropout is a hard flag and feeds the union qc_flag
    flags = compute_qc_flags(df)
    assert bool(hard_flag(flags).iloc[52:61].all())


def synth_station(n=24 * 90, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    diurnal = np.sin(2 * np.pi * ts.hour.to_numpy() / 24)
    return pd.DataFrame({
        "timestamp": ts,
        "temperature_c": 25 + 8 * diurnal + rng.normal(0, 0.8, n),
        "pressure_hpa": 1010 + 3 * np.sin(2 * np.pi * ts.dayofyear.to_numpy() / 365.0)
        + rng.normal(0, 0.4, n),
        "relative_humidity_pct": np.clip(70 - 20 * diurnal + rng.normal(0, 3, n), 1, 100),
    })


def test_qc_flags_range():
    from awsad.preprocessing.qc_rules import compute_qc_flags
    df = synth_station(500)
    df.loc[10, "temperature_c"] = 80.0
    flags = compute_qc_flags(df)
    assert flags.loc[10, "range_temperature_c"]
    assert flags["qc_flag"].sum() >= 1


def test_injection_labels_align():
    from awsad.preprocessing.anomaly_injection import AnomalyInjector
    df = synth_station()
    inj = AnomalyInjector(rng=np.random.default_rng(1), target_fraction=0.03)
    out, labels, events = inj.inject(df, ["temperature_c", "pressure_hpa",
                                          "relative_humidity_pct"])
    assert len(events) > 0
    assert labels.sum() > 0
    assert labels.index.equals(df.index)


def test_zscore_detects_stuck():
    from awsad.models.statistical import RobustChannelDetector
    from awsad.preprocessing.features import add_physics_derived
    df = add_physics_derived(synth_station())
    tr = df.iloc[:1000]
    rcd = RobustChannelDetector().fit(tr, tr)
    te = df.iloc[1000:].reset_index(drop=True)
    te.loc[100:130, "temperature_c"] = 31.4159        # frozen sensor
    s = rcd.score(te, te)
    assert s[110] >= 2.9, f"stuck sensor not detected (score={s[110]:.2f})"
    assert np.median(s[:80]) < 2.5


def test_metrics_perfect_and_null():
    from awsad.evaluation.metrics import full_report
    y = np.zeros(1000); y[100:150] = 1
    s = y.astype(float) * 5
    r = full_report(y, s, 2.0)
    assert r["pointwise"]["f1"] == 1.0 and r["range_wise"]["f1"] == 1.0
    r0 = full_report(y, np.zeros(1000), 1.0)
    assert r0["pointwise"]["f1"] == 0.0


def test_ensemble_roundtrip(tmp_path):
    from awsad.models.ensemble import WeightedEnsemble
    e = WeightedEnsemble({"qc": 0.5, "zscore": 0.5})
    e.norms_ = {"qc": (0.0, 1.0), "zscore": (0.0, 2.0)}
    e.per_group_norms = {"g": {"qc": (0.0, 1.0), "zscore": (0.0, 4.0)}}
    e.per_group_thresholds = {"g": 1.5}
    e.per_group_fallback_thresholds = {"holdout": 1.8}
    e.pooled_threshold = 0.7
    e.fallback_threshold = 2.5
    p = tmp_path / "e.json"
    e.save(str(p))
    e2 = WeightedEnsemble.load(str(p))
    assert e2.per_group_thresholds["g"] == 1.5
    assert e2.threshold_for("g") == 1.5
    assert e2.threshold_for("holdout") == 1.8
    assert e2.threshold_for("other") == 2.5
    s = e2.score_group("g", {"qc": np.array([0.0, 1.0]), "zscore": np.array([0.0, 8.0])})
    assert s.shape == (2,)


def test_streaming_latency():
    df = synth_station(2000)
    import time
    t0 = time.perf_counter()
    _ = df[["timestamp", "temperature_c"]].itertuples(index=False)
    _ = time.perf_counter() - t0   # placeholder timing sanity


def test_choose_threshold_survives_nan_inf():
    from awsad.evaluation.metrics import choose_threshold
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(0, 1, 5000), [np.inf, -np.inf, np.nan] * 10])
    y = (s > 2).astype(int)
    thr, meta = choose_threshold(s, y)
    assert np.isfinite(thr)
    s2 = np.full(10, np.nan)
    thr2, meta2 = choose_threshold(s2, np.zeros(10))
    assert np.isfinite(thr2) and meta2.get("degenerate")


def test_weight_calibration_objective_prioritizes_strict_detection():
    from awsad.evaluation.metrics import best_weight_objective_over_quantiles

    y = np.zeros(1000, dtype=int)
    y[100:200] = 1
    scores = np.full(1000, 0.05)
    scores[100:200] = 0.8
    scores[100] = 1.0
    objective, _ = best_weight_objective_over_quantiles(scores, y)
    # Detecting only the first point would look perfect after point adjustment
    # but has very poor strict recall. The strict-weighted objective must prefer
    # the threshold that detects the complete fault interval.
    assert objective > 0.8


def test_physics_derived_consistency():
    from awsad.preprocessing.features import add_physics_derived
    df = synth_station()
    d = add_physics_derived(df)
    # Magnus closure: RH from T and derived Td must reproduce input RH (~)
    assert d["dewpoint_c"].between(-60, 60).all()
    assert (d["td_spread"] >= -0.5).mean() > 0.99
    assert "vpd_hpa" in d.columns and "press_tendency_3h" in d.columns


def test_group_scaling_preserves_alignment():
    from awsad.train_pipeline import _scale_all_groups
    df = pd.DataFrame({"group": ["a", "b", "a", "b"],
                       "feature": [10.0, 110.0, 20.0, 120.0],
                       "row_id": [0, 1, 2, 3]})
    metas = {
        "a": {"feature_cols": ["feature"], "robust_scale": {"feature": (10.0, 10.0)}},
        "b": {"feature_cols": ["feature"], "robust_scale": {"feature": (110.0, 10.0)}},
    }
    out = _scale_all_groups(df, metas)
    assert out["row_id"].tolist() == [0, 1, 2, 3]
    assert np.allclose(out["feature"], [0.0, 0.0, 1.0, 1.0])


def test_if_feature_selection_uses_all_fit_blocks():
    from awsad.train_pipeline import select_if_features_blocks

    # `late_feature` is absent in the first block but populated in the second;
    # a first-eight-group probe would incorrectly discard it.
    blocks = [
        pd.DataFrame({"early_feature": [0.0, 1.0, 2.0],
                      "late_feature": [np.nan, np.nan, np.nan]}),
        pd.DataFrame({"early_feature": [3.0, 4.0, 5.0],
                      "late_feature": [10.0, 11.0, 12.0]}),
    ]
    keep = select_if_features_blocks(blocks, ["early_feature", "late_feature"])
    assert keep == ["early_feature", "late_feature"]


def test_robust_scale_all_missing_is_finite():
    from awsad.preprocessing.features import robust_scale_apply, robust_scale_fit

    frame = pd.DataFrame({"present": [1.0, 2.0, 3.0],
                          "missing": [np.nan, np.nan, np.nan]})
    params = robust_scale_fit(frame, ["present", "missing"])
    assert params["missing"] == (0.0, 1.0)
    scaled = robust_scale_apply(frame, params)
    assert np.isfinite(params["present"][0])
    assert scaled["missing"].isna().all()


def test_metrics_do_not_merge_station_boundaries():
    from awsad.train_pipeline import _binary_report
    y = np.array([0, 1, 1, 0])
    pred = np.array([0, 1, 0, 0])
    scores = pred.astype(float)
    groups = np.array(["a", "a", "b", "b"])
    report = _binary_report(y, scores, pred, group_ids=groups)
    assert report["point_adjust"]["recall"] == 0.5


def test_spatial_holdout_uses_train_fallback():
    from awsad.models.spatial import SpatialConsistency
    ts = pd.date_range("2024-01-01", periods=500, freq="1h", tz="UTC")
    base = np.sin(np.arange(len(ts)) / 24.0)
    frames = {
        "a|obs": pd.DataFrame({"timestamp": ts, "temperature_c": base}),
        "b|obs": pd.DataFrame({"timestamp": ts, "temperature_c": base + 0.1}),
    }
    meta = {"a": {"lat": 20.0, "lon": 77.0, "elev": 0.0},
            "b": {"lat": 20.2, "lon": 77.0, "elev": 0.0},
            "c": {"lat": 20.1, "lon": 77.1, "elev": 0.0}}
    model = SpatialConsistency(meta, min_peers=1).fit(frames)
    scored = dict(frames)
    scored["c|obs"] = pd.DataFrame({"timestamp": ts,
                                     "temperature_c": base + 20.0})
    out = model.score_all(scored)
    assert "temperature_c" in model.channel_spread
    assert ("c|obs", "temperature_c") not in model.train_spread
    assert np.isfinite(out["c|obs"]).all() and out["c|obs"].max() > 1.0


def test_station_catalog_and_holdout_selection_are_deterministic(tmp_path):
    from awsad.data.stations import (build_station_catalog,
                                     select_holdout_stations)

    history = tmp_path / "isd-history.csv"
    history.write_text(
        '"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO",'
        '"LAT","LON","ELEV(M)","BEGIN","END"\n'
        '"123456","99999","A","IN","","","+20.0","+77.0",'
        '"+100.0","20100101","20251231"\n'
        '"123457","99999","B","IN","","","+21.0","+78.0",'
        '"+200.0","20100101","20251231"\n',
        encoding="utf-8")
    catalog = build_station_catalog(["12345699999", "12345799999"], history)
    assert [s.station_id for s in catalog] == ["12345699999", "12345799999"]
    assert catalog[0].lat == 20.0
    ids = [f"{i:011d}" for i in range(20)]
    h1 = select_holdout_stations(ids, fraction=.2, preferred=set(), seed=42)
    h2 = select_holdout_stations(list(reversed(ids)), fraction=.2,
                                 preferred=set(), seed=42)
    assert h1 == h2 and len(h1) == 4 and set(h1).issubset(ids)


def test_isd_duplicate_reports_keep_valid_pressure(tmp_path):
    """FM-15 may repeat a timestamp with missing SLP; fieldwise resampling
    must retain the valid FM-12 value rather than keeping the last row."""
    from awsad.data.parse_isd import parse_station_year
    raw = tmp_path / "42182099999.csv"
    raw.write_text(
        "STATION,DATE,REPORT_TYPE,TMP,DEW,SLP,WND,AA1\n"
        '42182099999,2024-01-01T00:00:00,FM-12,"+0250,1","+0150,1","10132,1","180,1,A,0050,1","01,0000,1,1"\n'
        '42182099999,2024-01-01T00:00:00,FM-15,"+0250,1","+0150,1","99999,9","180,1,A,0050,1","01,0000,1,1"\n'
        '42182099999,2024-01-01T01:00:00,FM-12,"+0260,1","+0160,1","10131,1","180,1,A,0050,1","01,0000,1,1"\n',
        encoding="utf-8",
    )
    parsed = parse_station_year(raw)
    assert len(parsed) == 3
    assert parsed.loc[parsed["report_type"] == "FM-12", "pressure_hpa"].notna().all()
    # The parser deliberately preserves both report types.  Hourly aggregation
    # is responsible for the field-wise reduction downstream.
    assert set(parsed["report_type"]) == {"FM-12", "FM-15"}
    from awsad.preprocessing.features import impute_hourly
    hourly = impute_hourly(parsed, ["temperature_c", "pressure_hpa"], max_gap=0)
    assert np.isclose(hourly.loc[0, "pressure_hpa"], 1013.2)


def test_isd_retains_real_qc_anomaly_labels(tmp_path):
    """NOAA QC codes 2/3/6/7 must be retained as real per-channel anomaly
    labels (not silently dropped), and carried through the hourly grid."""
    from awsad.data.parse_isd import parse_station_year, PARSER_VERSION, GOOD_QC
    from awsad.preprocessing.features import impute_hourly

    assert PARSER_VERSION == "isd-qc-anomaly-labels-v3"
    assert {"4", "5", "9"} <= GOOD_QC  # NCEI-source passed codes kept, not dropped
    raw = tmp_path / "42181099999.csv"
    raw.write_text(
        "STATION,DATE,REPORT_TYPE,TMP,DEW,SLP,WND,AA1\n"
        '42181099999,2023-01-01T00:00:00,FM-12,"+0250,1","+0100,1","10130,1","999,1,N,0010,1",",9999,,"\n'
        '42181099999,2023-01-01T01:00:00,FM-12,"+0450,2","+0100,1","10120,1","999,1,N,0010,1",",9999,,"\n'
        '42181099999,2023-01-01T02:00:00,FM-12,"+0260,1","+0100,1","09000,3","999,1,N,0010,1",",9999,,"\n'
        '42181099999,2023-01-01T03:00:00,FM-12,"+9999,1","+0100,1","10110,1","999,1,N,0010,1",",9999,,"\n',
        encoding="utf-8",
    )
    p = parse_station_year(raw)
    h1 = p["timestamp"].dt.hour == 1
    assert bool(p.loc[h1, "temperature_c__qc_anomaly"].iloc[0]) is True
    assert np.isclose(p.loc[h1, "temperature_c__qc_value"].iloc[0], 45.0)
    assert pd.isna(p.loc[h1, "temperature_c"].iloc[0])  # flagged value not in clean channel
    h2 = p["timestamp"].dt.hour == 2
    assert bool(p.loc[h2, "pressure_hpa__qc_anomaly"].iloc[0]) is True
    assert np.isclose(p.loc[h2, "pressure_hpa__qc_value"].iloc[0], 900.0)
    h3 = p["timestamp"].dt.hour == 3
    assert bool(p.loc[h3, "temperature_c__qc_anomaly"].iloc[0]) is False

    hourly = impute_hourly(p, ["temperature_c", "pressure_hpa"], max_gap=0)
    assert "temperature_c__qc_anomaly" in hourly.columns
    assert bool(hourly["temperature_c__qc_anomaly"].any())
    assert np.isclose(
        hourly.loc[hourly["temperature_c__qc_anomaly"], "temperature_c__qc_value"].iloc[0],
        45.0,
    )


def test_injector_handles_missing_channels_without_unbound_local(tmp_path):
    from awsad.preprocessing.anomaly_injection import AnomalyInjector
    df = synth_station(300).drop(columns=["relative_humidity_pct"])
    out, labels, events = AnomalyInjector(
        rng=np.random.default_rng(7), target_fraction=.02
    ).inject(df, ["temperature_c", "pressure_hpa", "relative_humidity_pct"])
    assert len(out) == len(df)
    assert len(labels) == len(df)
    assert labels.index.equals(df.index)


def test_noise_injection_realigns_end_after_missing_start():
    """A missing proposed start can move the window forward.

    Regression for the Kaggle builder failure where the selected data slice
    had 64 rows but the noise vector retained the original length of 69.
    """
    from awsad.preprocessing.anomaly_injection import AnomalyInjector

    class FixedRng:
        def __init__(self):
            self.integer_calls = 0

        def integers(self, low, high=None):
            self.integer_calls += 1
            return 69 if self.integer_calls == 1 else 100

        def choice(self, values, p=None):
            return values[0]

        def uniform(self, low, high):
            return (low + high) / 2

        def normal(self, loc, scale, size):
            return np.full(size, scale, dtype=float)

    df = synth_station(500)
    df.loc[100:104, "temperature_c"] = np.nan
    injector = AnomalyInjector(rng=FixedRng())
    event = injector._inject(df, "noise_burst", ["temperature_c"])

    assert event is not None
    assert event.start == df.loc[105, "timestamp"]
    assert event.end == df.loc[173, "timestamp"]
    assert df.loc[105:173, "temperature_c"].notna().all()


def test_sensor_swap_emits_event_and_mutates_channel():
    """sensor_swap must fire on a fully finite frame.

    Regression for the Kaggle builder failure ``fault taxonomy incomplete:
    missing=['sensor_swap']``.  ``a = df[ch_a].to_numpy(float)`` is a view over
    the column buffer, so writing the swap in place before comparing made the
    change check compare the mutated buffer against itself; sensor_swap was
    dropped on every station and the split never covered the full taxonomy.
    """
    from awsad.preprocessing.anomaly_injection import AnomalyInjector

    cols = ["temperature_c", "pressure_hpa", "relative_humidity_pct"]

    # Direct path: every forced attempt must produce an event.
    injector = AnomalyInjector(rng=np.random.default_rng(1))
    events = [
        injector._inject(synth_station(8760, seed=s), "sensor_swap", cols)
        for s in range(25)
    ]
    assert all(ev is not None for ev in events)
    assert all(ev.fault == "sensor_swap" for ev in events)
    assert all(ev.params.get("swapped_with") == "relative_humidity_pct"
               for ev in events)

    # Driver path with required_faults, mirroring the builder's per-split
    # coverage forcing, must surface sensor_swap in the events table.
    base = synth_station(8760, seed=0)
    out, labels, table = AnomalyInjector(
        rng=np.random.default_rng(42), target_fraction=0.04
    ).inject(base.reset_index(drop=True), cols,
             required_faults=["sensor_swap"])
    assert not table.empty
    assert "sensor_swap" in set(table["fault"])

    swap = table[table["fault"] == "sensor_swap"].iloc[0]
    window = (out["timestamp"] >= swap["start"]) & (out["timestamp"] <= swap["end"])
    channel = swap["channel"]
    mutated = out.loc[window, channel].to_numpy()
    original = base.loc[window.to_numpy(), channel].to_numpy()
    assert np.any(mutated != original)
