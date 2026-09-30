"""Serve the SkyGuard React frontend + JSON API with uvicorn (this replaces the old
Streamlit dashboard). Build the SPA first, then run this:

    npm --prefix frontend run build
    python scripts/serve_dashboard.py [port]      # default port 8501

The Starlette app (app/api.py) reads the same verified artifacts as before; set
SKYGUARD_ARTIFACTS / SKYGUARD_INDIAN_DEMO to point at other bundles.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")


def main() -> None:
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    uvicorn.run("app.api:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
