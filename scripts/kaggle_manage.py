"""Operate the team's private Kaggle jobs without printing or bundling credentials.

The credential file is read locally; only authentication to Kaggle uses it.
Dataset downloads and preparation happen in the online CPU builder on Kaggle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def api_client(credentials: Path):
    # The desktop's bundled Python can use the pure-Python Kaggle client already
    # installed in .venv when that environment's executable is unavailable.
    local_packages = ROOT / ".venv/Lib/site-packages"
    if local_packages.is_dir():
        sys.path.append(str(local_packages))
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    if credentials.exists():
        secret = json.loads(credentials.read_text(encoding="utf-8"))
        if "username" not in secret or "key" not in secret:
            raise ValueError("Expected a Kaggle API credential file with username and key")
        os.environ["KAGGLE_USERNAME"] = secret["username"]
        os.environ["KAGGLE_KEY"] = secret["key"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def safe_error(exc: Exception) -> str:
    message = str(exc)
    for name in ("KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return f"{type(exc).__name__}: {message[:1800]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "publish-dataset", "publish-processed",
                                              "push-builder", "push-training",
                                              "status", "logs"])
    parser.add_argument("--credentials", type=Path, default=ROOT / "kaggle-krishna.json")
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--directory", type=Path, help="processed release directory containing only its ZIP and dataset-metadata.json")
    args = parser.parse_args()
    api = api_client(args.credentials)
    paths = {"builder": ROOT / "kaggle_builder", "training": ROOT / "kaggle"}
    meta = {k: json.loads((p / "kernel-metadata.json").read_text()) for k, p in paths.items()}
    if args.command in {"publish-dataset", "publish-processed"}:
        staging = ROOT / "data" / "kaggle_staging" if args.command == "publish-dataset" else args.directory
        if staging is None:
            raise ValueError("publish-processed requires --directory")
        if args.command == "publish-processed":
            import zipfile
            contents = list(staging.iterdir())
            archives = [p for p in contents if p.suffix == ".zip"]
            if len(archives) != 1 or {p.name for p in contents} != {archives[0].name, "dataset-metadata.json"}:
                raise ValueError("processed upload directory must contain exactly its release ZIP and metadata")
            with zipfile.ZipFile(archives[0]) as archive:
                manifest_names = [n for n in archive.namelist() if n == "MANIFEST.json"]
                if not manifest_names:
                    raise ValueError("processed ZIP lacks a root release manifest")
                release = json.loads(archive.read("MANIFEST.json"))
                if release.get("bundle_role") != "processed_training_bundle" or release.get("training_data_policy") != "real_observations_only":
                    raise ValueError("only real-observation processed releases may be published")
        if not (staging / "dataset-metadata.json").exists():
            raise FileNotFoundError(f"missing {staging / 'dataset-metadata.json'}; run prepare_kaggle_upload.py")
        response = api.dataset_create_version(
            str(staging),
            "Verified-scope SURFRAD observations and causal CPU training; no synthetic observations or fault labels",
            quiet=False,
            convert_to_csv=False,
            dir_mode="skip",
        )
        if getattr(response, "error", None) or str(getattr(response, "status", "")).lower() == "error":
            raise RuntimeError("Kaggle refused dataset publication: " + str(getattr(response, "error", response)))
        print(json.dumps({"result": response}, default=str))
    elif args.command == "logs":
        kernel = args.kernel
        if not kernel:
            raise ValueError("logs requires --kernel builder or --kernel training")
        info = meta.get(kernel)
        if not info:
            raise ValueError(f"unknown kernel kind: {kernel}")
        print(api.kernels_logs(info["id"]))
    elif args.command in ("inspect", "status"):
        for kind, info in meta.items():
            if args.kernel and args.kernel != kind:
                continue
            try:
                status = api.kernels_status(info["id"])
                print(json.dumps({"kernel": info["id"], "status": status}, default=str))
            except Exception as exc:
                print(json.dumps({"kernel": info["id"], "error": safe_error(exc)}))
        if args.command == "inspect":
            datasets = meta["builder"].get("dataset_sources", [])
            for dataset in datasets:
                try:
                    print(json.dumps(
                        {"builder_source_dataset": dataset,
                         "status": api.dataset_status(dataset)}, default=str))
                    files = api.dataset_list_files(dataset)
                    print(json.dumps({"dataset_files": str(files)[:4000]}))
                except Exception as exc:
                    print(json.dumps({"dataset": dataset, "error": safe_error(exc)}))
            print(json.dumps({
                "training_kernel_sources": meta["training"].get("kernel_sources", [])
            }))
    else:
        kind = args.command.removeprefix("push-")
        info = meta[kind]
        if not info.get("is_private"):
            raise ValueError("SkyGuard jobs must stay private")
        if kind == "builder" and (info.get("enable_gpu") or not info.get("enable_internet")):
            raise ValueError("Builder must use CPU with Internet on")
        if kind == "training" and info.get("enable_internet"):
            raise ValueError("Training must keep Internet off")
        print(json.dumps({"result": api.kernels_push(str(paths[kind]))}, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        sys.exit(1)
