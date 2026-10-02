"""Serve the SkyGuard React frontend + JSON API with uvicorn (this replaces the old
Streamlit dashboard). Build the SPA first, then run this:

    npm --prefix frontend run build
    python scripts/serve_dashboard.py [port]      # default port 8501

The Starlette app (app/api.py) reads the same verified artifacts as before; set
SKYGUARD_ARTIFACTS / SKYGUARD_INDIAN_DEMO to point at other bundles.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", type=int, default=8501)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run("app.api:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
