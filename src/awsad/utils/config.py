"""Config loading + project paths. Dependency-light (PyYAML with a tiny fallback)."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def project_root() -> Path:
    return Path(os.environ.get("AWSAD_ROOT", PROJECT_ROOT))


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = project_root() / path
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:  # pragma: no cover - minimal fallback
        # Very small fallback parser for our flat-ish config; only used if PyYAML
        # is unavailable (never the case on Kaggle).
        raise RuntimeError("PyYAML is required: pip install pyyaml")


def resolve(root: Path, cfg: dict) -> dict:
    """Resolve relative data paths against the project root."""
    out = dict(cfg)
    for k, v in cfg.get("paths", {}).items():
        out[f"path_{k}"] = root / v if not Path(v).is_absolute() else Path(v)
    return out
