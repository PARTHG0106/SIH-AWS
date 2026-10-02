# Architecture — SkyGuard AI

> Historical architecture of the superseded injected/hourly pipeline. This is
> not a description of the active September 30 system or evidence for its
> performance. See `README.md`, `docs/LIVE_DETECTION_20260930.md` and
> `docs/SIH_COMPLETION_PLAN.md` for the current measured/synthetic separation.

```
                     ┌─────────────────────────────────────────────────────┐
                     │                 DATA FOUNDATION                     │
                     │  NOAA ISD (real stations) · Open-Meteo twins        │
                     │  NCPOR/IMD AWS · NAB benchmark · SMAP/MSL (Kaggle)  │
                     └────────────┬────────────────────────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │   PREPARATION LAYER     │  hourly grid · QC denoise
                     │   prepare_dataset.py    │  Magnus RH·Td physics closure
                     │                         │  fault INJECTION + labels
                     └────────────┬────────────┘  train|val|test splits
                                  │                      + holdout stations
   ┌──────────┬──────────┬────────▼───────┬────────────┬───────────┬────────────┐
   │ PHYSICS  │ ROBUST   │ ISOLATION      │ DEEP HEADS │ DEEP HEAD │  SPATIAL   │
   │ QC RULES │ CHANNEL  │ FOREST         │ LSTM-AE    │ TRANSFORM │ BUDDY-CHECK│
   │ (limits, │ DETECTOR │ (windowed      │ 24h +168h  │ AE 48h +  │ (titanlib- │
   │ steps,   │ level+   │ features)      │            │ FORECAST  │ style)     │
   │ spikes,  │ dynamics │                │            │ ER LSTM   │            │
   │ flatline)│          │                │            │           │            │
   └────┬─────┴────┬─────┴───────┬────────┴─────┬──────┴─────┬─────┴──────┬─────┘
        │          │             │              │            │            │
        │    per-station normalisation (median, q99 from clean train)     │
        └──────────┴─────────────┴──────────────┴────────────┴────────────┘
                                  │
                ┌─────────────────▼─────────────────┐
                │  CALIBRATED ENSEMBLE (weighted)   │  weights learned on
                │  + per-station supervised thr     │  labeled val split
                │  + EVT/POT fallback for new ones  │
                └─────────────────┬─────────────────┘
                                  │
        ┌─────────────┬───────────┼──────────────┬───────────────┐
        ▼             ▼           ▼              ▼               ▼
   Alerts w/      Corrected    Sensor-health   Streamlit       Streaming
   confidence,    values       index +         dashboard       mode (µs/obs,
   reasons,       (fusion)     maintenance     (5 pages)       ESP32-class)
   root cause
```

## Design principles

1. **PS-honest inputs**: only Temperature, Pressure, Relative Humidity are read
   by detectors. Dew point, VPD, pressure tendency are *derived* from those
   three (Magnus / Tetens / 3h tendency) and documented as such.
2. **No leakage**: climatology, robust scalers and score norms use clean history;
   model weights and supervised thresholds use validation labels only for the
   non-holdout groups. Holdout stations are excluded from global IF/deep fitting
   and all label-based calibration. Their local clean-history baseline and POT
   threshold are reported explicitly as unsupervised adaptation.
3. **Per-station normalisation**: every network node gets its own scale and
   threshold — "what is extreme at Jaisalmer is normal; the same value in
   Kodaikanal is a fault" is encoded by construction.
4. **Defence in depth**: independent physical reasoning (QC, physics identities),
   classical ML (IF), deep reconstruction & forecasting heads, and network-level
   cross-checks. A fault has to hide from all of them to be missed.
5. **Streaming contract**: the bounded-state streaming detector uses the QC and
   robust-statistical components from the batch system with a separately
   calibrated fast-path threshold. The full IF/deep/spatial ensemble runs at the
   hub or in periodic micro-batches; the two tiers are not claimed to be bitwise
   identical.

## Metric protocol (rigorous by design)
- point-wise P/R/F1 (strict), point-adjusted F1 (reported WITH the caveat of
  Kim et al. 2022), range-wise P/R/F1 (Tatbul et al. 2018), PR/ROC-AUC,
  mean detection latency.
- Per-fault-type recall table (9 injected fault categories) + leave-one-out
  ablation + unseen-station generalisation split.

## Failure-mode coverage map

| Injected fault | Primary detector | Backup |
|---|---|---|
| spike | QC step/spike + z-step | deep heads |
| stuck (frozen) | z-stuckness (raw-domain) | QC flatline, AE |
| drift | AE/forecaster residual growth | z-level, QC step |
| bias | z-level, spatial buddy | AE, QC consistency |
| dropout | z-missing hard 4.0 | QC, IF masks |
| noise burst | z-noisiness (rolling σ) | IF, AE |
| clipping | plateau detection | AE, z-level |
| scale error | weekly rolling-σ ratio | AE |
| sensor swap | cross-channel consistency (Td closure, spatial) | IF |

## Deployment tiers
- **Edge (ESP32-S3):** range/step/flatline QC + clipped z (µs/obs, no floats needed beyond float32)
- **Hub (this repo):** full ensemble + explainability + health index (bench: ~51 µs/obs single-core CPU)
- **Centre:** batch retraining on Kaggle GPU; artifact refresh cycle.
