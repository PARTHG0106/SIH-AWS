"""Warm, natural visual theme for the SkyGuard dashboards — light and dark aware.

Presentation only: nothing here reads observations, computes scores, or changes a
label. The observed/source series is carried by high-contrast ink (dominant in
either mode); the two accent hues (terracotta, teal) are validated colorblind-safe
against each other; alerts use a reserved warm-crimson status hue with ring markers
and reason tooltips, never colour alone.
"""
from __future__ import annotations

import altair as alt
import streamlit as st

PALETTES = {
    "light": {
        "mode": "light",
        "bg_top": "#FBF6EF", "bg_bottom": "#F1E7D6",
        "surface": "#FFFDF9", "surface_soft": "#FBF3E7",
        "border": "#EADCC6", "ink": "#2E2822", "muted": "#6E6053", "faint": "#9C8B77",
        "accent": "#BC5A2E", "accent2": "#0E7E70", "alert": "#B23A48",
        "grid": "#ECE1CF", "axis": "#8A7B69", "shadow": "rgba(122, 92, 52, 0.10)",
        "area_top": "rgba(188, 90, 46, 0.20)", "area_bottom": "rgba(188, 90, 46, 0.0)",
    },
    "dark": {
        "mode": "dark",
        "bg_top": "#221E19", "bg_bottom": "#17130F",
        "surface": "#2A2520", "surface_soft": "#332C23",
        "border": "#3C3429", "ink": "#F1E8DA", "muted": "#B4A491", "faint": "#8A7B67",
        "accent": "#E38C5F", "accent2": "#48B6A4", "alert": "#F0899A",
        "grid": "#3A332A", "axis": "#9C8E7B", "shadow": "rgba(0, 0, 0, 0.38)",
        "area_top": "rgba(227, 140, 95, 0.26)", "area_bottom": "rgba(227, 140, 95, 0.0)",
    },
}


def theme_type() -> str:
    """Resolve the viewer's active Streamlit theme ('light' or 'dark')."""
    try:
        value = getattr(getattr(st, "context", None), "theme", None)
        return "dark" if getattr(value, "type", None) == "dark" else "light"
    except Exception:
        return "light"


def palette() -> dict:
    return PALETTES[theme_type()]


_VAR_KEYS = ("bg_top", "bg_bottom", "surface", "surface_soft", "border", "ink", "muted",
             "faint", "accent", "accent2", "alert", "grid", "axis", "shadow")


def _vars(p: dict) -> str:
    return "".join(f"--sg-{key.replace('_', '-')}:{p[key]};" for key in _VAR_KEYS)


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');
[data-testid="stAppViewContainer"], .stApp {
  background: radial-gradient(1100px 560px at 80% -10%, var(--sg-surface-soft), transparent 60%),
              linear-gradient(168deg, var(--sg-bg-top), var(--sg-bg-bottom)) fixed;
  color: var(--sg-ink); font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2.2rem; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif; color: var(--sg-ink); letter-spacing: -0.015em; }
h1 { font-weight: 600; font-size: 2.5rem; line-height: 1.12; }
h2 { font-weight: 600; }
a, a:visited { color: var(--sg-accent); }
[data-testid="stSidebar"] { background: var(--sg-surface); border-right: 1px solid var(--sg-border); }
[data-testid="stSidebar"] h1 { font-size: 1.5rem; }
[data-testid="stMetric"] { background: var(--sg-surface); border: 1px solid var(--sg-border);
  border-radius: 16px; padding: 15px 18px 13px; box-shadow: 0 1px 3px var(--sg-shadow); }
[data-testid="stMetricValue"] { font-family: 'Fraunces', serif; font-weight: 600; color: var(--sg-ink); }
[data-testid="stMetricLabel"] { color: var(--sg-muted); font-weight: 500; }
[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--sg-border); background: transparent; }
[data-baseweb="tab"] { color: var(--sg-muted); font-weight: 500; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--sg-accent); }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { background: var(--sg-accent); }
[data-testid="stVegaLiteChart"] { background: var(--sg-surface); border: 1px solid var(--sg-border);
  border-radius: 16px; padding: 12px 14px 6px; box-shadow: 0 1px 3px var(--sg-shadow); }
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--sg-border); }
.stButton button, .stDownloadButton button, [data-testid="stSidebar"] button {
  border-radius: 10px; border: 1px solid var(--sg-border); background: var(--sg-surface-soft);
  color: var(--sg-ink); font-weight: 500; transition: border-color .15s ease; }
.stButton button:hover, [data-testid="stSidebar"] button:hover { border-color: var(--sg-accent); color: var(--sg-accent); }
.stDownloadButton button { background: var(--sg-accent); color: #fff; border-color: transparent; }
[data-testid="stExpander"] { border: 1px solid var(--sg-border); border-radius: 14px;
  background: var(--sg-surface); overflow: hidden; }
[data-testid="stExpander"] summary:hover { color: var(--sg-accent); }
[data-testid="stDataFrame"] { border-radius: 12px; border: 1px solid var(--sg-border); }
[data-testid="stMetricLabel"] p { white-space: normal; overflow: visible; text-overflow: clip; }
@media (max-width: 1100px) {
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
    width: 100% !important; min-width: 0 !important; }
  [data-testid="stMetric"] { padding: 12px; }
}
/* __CSS_TAIL__ */
"""


def inject_css(p: dict) -> None:
    """Inject the warm/natural design system for the viewer's active light/dark mode."""
    st.markdown(f"<style>:root{{{_vars(p)}}}{_CSS}</style>", unsafe_allow_html=True)


def configure_chart(chart: alt.LayerChart | alt.Chart, p: dict, *, height: int = 240):
    """Apply the shared warm chart theme: transparent surface, recessive axes/grid."""
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(gridColor=p["grid"], gridOpacity=0.65, domainColor=p["grid"],
                            tickColor=p["grid"], labelColor=p["axis"], titleColor=p["muted"],
                            labelFont="Inter", titleFont="Inter", labelFontSize=11,
                            titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelColor=p["muted"], titleColor=p["muted"], labelFont="Inter",
                              labelFontSize=12, symbolStrokeWidth=2.5, symbolSize=130, orient="top")
            .configure_axisY(gridDash=[2, 3]))
