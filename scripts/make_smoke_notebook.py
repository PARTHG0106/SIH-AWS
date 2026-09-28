"""Generate a fast smoke variant of the training notebook: same code path and
bootstrap, but a handful of groups and 1-2 tiny epochs so a full end-to-end run
(incl. forecaster score_series + ensemble + eval) finishes in ~10 min on the
RTX Pro 6000.  Writes kaggle_smoke/aws_anomaly_training_smoke.ipynb + metadata.
"""
import json, pathlib, copy

ROOT = pathlib.Path(__file__).resolve().parents[1]
full = json.loads((ROOT / "kaggle" / "aws_anomaly_training.ipynb").read_text(encoding="utf-8"))
nb = copy.deepcopy(full)

SMOKE_HEAD = '''from awsad.train_pipeline import run_pipeline

# SMOKE: tiny subset + few epochs to validate the whole pipeline end-to-end
# (featurize -> classical -> deep AEs -> forecaster score_series -> ensemble ->
# eval) in ~10 min on the RTX Pro 6000 before committing to the full run.
NOAA_GROUPS = NOAA_GROUPS[:12]
print(f"SMOKE: restricted to {len(NOAA_GROUPS)} NOAA groups")
DEEP_CFG = {
    "lstm_ae_24":  {"epochs": 2, "batch_size": 512, "windows_per_epoch": 8192,
                    "hidden": 64, "layers": 2, "dropout": 0.10,
                    "balance_groups": True, "loss": "huber", "val_windows": 2048},
    "lstm_ae_168": {"epochs": 2, "batch_size": 256, "windows_per_epoch": 8192,
                    "hidden": 64, "layers": 2, "dropout": 0.10,
                    "balance_groups": True, "loss": "huber", "val_windows": 2048},
    "tx_ae":       {"epochs": 2, "batch_size": 256, "windows_per_epoch": 8192,
                    "d_model": 64, "nhead": 4, "layers": 2, "ff": 128,
                    "bottleneck": 32, "dropout": 0.10,
                    "balance_groups": True, "loss": "huber", "val_windows": 2048},
    "forecaster":  {"epochs": 2, "batch_size": 512, "windows_per_epoch": 8192,
                    "hidden": 64, "layers": 2, "dropout": 0.10,
                    "balance_groups": True, "loss": "huber", "val_windows": 2048},
}
t0 = time.time()
'''

patched = False
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if "run_pipeline(dfs, NOAA_GROUPS" in s and "DEEP_CFG" in s:
        tail = s.split("t0 = time.time()", 1)[1]
        c["source"] = SMOKE_HEAD + tail
        patched = True
        break
assert patched, "could not find the run_pipeline/DEEP_CFG cell"

out_dir = ROOT / "kaggle_smoke"
out_dir.mkdir(exist_ok=True)
(out_dir / "aws_anomaly_training_smoke.ipynb").write_text(
    json.dumps(nb, indent=1), encoding="utf-8")
(out_dir / "kernel-metadata.json").write_text(json.dumps({
    "id": "krishnagupta02468/skyguard-ai-training-smoke-sih26073",
    "title": "SkyGuard AI - Training SMOKE (SIH26073)",
    "code_file": "aws_anomaly_training_smoke.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "machine_shape": "NvidiaRtxPro6000",
    "dataset_sources": ["krishnagupta02468/skyguard-sih26073-processed"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [],
    "model_sources": [],
    "category_ids": [],
}, indent=2), encoding="utf-8")
print("wrote", out_dir / "aws_anomaly_training_smoke.ipynb")
print("wrote", out_dir / "kernel-metadata.json")
