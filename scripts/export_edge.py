"""Edge packaging: INT8 dynamic-quantised models + size/latency report.

    python scripts/export_edge.py --artifacts artifacts_full
Produces artifacts/edge/*.ptz (quantised) and prints a size/latency table —
the numbers that back the Energy-Efficiency / Deployability claims.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def size_kb(path: Path) -> float:
    return path.stat().st_size / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts_smoke")
    args = ap.parse_args()
    art = Path(args.artifacts)
    out = art / "edge"; out.mkdir(exist_ok=True)

    rows = []
    for name in ("lstm_ae_24", "lstm_ae_168", "tx_ae", "forecaster"):
        p = art / f"{name}.pt"
        if not p.exists():
            continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        # rebuild generic module handles; quantise all LSTM/Linear layers
        if name.startswith("lstm"):
            from awsad.models.lstm_autoencoder import LSTMAEDetector
            det = LSTMAEDetector.load(str(p), device="cpu")
        elif name == "tx_ae":
            from awsad.models.transformer_ae import TransformerAEDetector
            det = TransformerAEDetector.load(str(p), device="cpu")
        else:
            from awsad.models.lstm_forecaster import ForecasterDetector
            det = ForecasterDetector.load(str(p), device="cpu")
        # LSTM-family: INT8 dynamic quantization. Transformer: FP16 CPU-safe.
        if name == "tx_ae":
            det.model.half()
            qmodel = det.model
            fmt = "fp16"
        else:
            qmodel = torch.quantization.quantize_dynamic(
                det.model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
            fmt = "int8"
        qpath = out / f"{name}_{fmt}.ptz"
        torch.save({"state_dict": qmodel.state_dict(), "cfg": det.cfg_model,
                    "feature_cols": det.feature_cols, "format": fmt}, qpath)

        n_feat = len(det.feature_cols)
        w = det.cfg_model.get("window", 24)
        x = torch.randn(1, w, n_feat)
        if fmt == "fp16":
            x = x.half()
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(50):
                    qmodel(x)
            dt = (time.perf_counter() - t0) / 50 * 1000
        except Exception as e:
            dt = float("nan"); print(f"  (latency probe failed for {name}: {e})")
        rows.append({"model": name, "format": fmt,
                     "fp32_kb": round(size_kb(p), 1),
                     "edge_kb": round(size_kb(qpath), 1),
                     "edge_ms_per_window_cpu": round(dt, 2)})

    for r in rows:
        print(f"  {r['model']:14s} {r['format']:5s} fp32={r['fp32_kb']:8.1f} KB  "
              f"edge={r['edge_kb']:7.1f} KB  cpu_latency={r['edge_ms_per_window_cpu']:6.2f} ms/window")
    print(f"\nquantised models -> {out}")


if __name__ == "__main__":
    main()
