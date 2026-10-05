"""
theme.py

Colour palette and Streamlit CSS injection for the dashboard, kept
separate from app.py so styling changes don't require going through
the benchmark/upload logic. Petrol blue primary, lime accent, warm
neutral background.
"""

import streamlit as st

COLOUR_BG = "#F6F9F7"
COLOUR_BG_SIDEBAR = "#E3EBE6"
COLOUR_PRIMARY = "#1B4F63"
COLOUR_PRIMARY_DARK = "#11333F"
COLOUR_TEXT = "#1E2420"
COLOUR_MUTED = "#5C6B60"
COLOUR_BORDER = "#DCE3DC"
COLOUR_ACCENT_LIME = "#8FAE3E"
COLOUR_BEST_HIGHLIGHT = "#E2EBB8"

PLOTLY_COLOURWAY = [
    "#1B4F63", "#8FAE3E", "#4E7A93", "#5C6B60",
    "#11333F", "#6B8F3E", "#3E6B7A", "#9AB56B",
]


def apply_theme():
    """Call once, right after st.set_page_config(), to inject the CSS block."""
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {COLOUR_BG};
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {COLOUR_BG_SIDEBAR};
        border-right: 1px solid {COLOUR_BORDER};
    }}
    h1, h2, h3 {{
        color: {COLOUR_PRIMARY_DARK};
        font-weight: 650;
    }}
    .app-subtitle {{
        color: {COLOUR_MUTED};
        font-size: 1.02rem;
        margin-top: -0.6rem;
        margin-bottom: 1.2rem;
    }}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {{
        color: {COLOUR_TEXT};
    }}
    [data-testid="stMarkdownContainer"] a {{
        color: {COLOUR_ACCENT_LIME};
        font-weight: 600;
    }}
    [data-testid="stMarkdownContainer"] a:hover {{
        color: {COLOUR_PRIMARY_DARK};
    }}
    .stButton > button {{
        background-color: {COLOUR_PRIMARY};
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.45rem 1.1rem;
        transition: background-color 0.15s ease;
    }}
    .stButton > button:hover {{
        background-color: {COLOUR_PRIMARY_DARK};
        color: #FFFFFF;
    }}
    [data-testid="stExpander"] summary {{
        color: {COLOUR_PRIMARY_DARK};
        font-weight: 600;
    }}
    hr {{
        border-color: {COLOUR_BORDER};
    }}
    [data-testid="stMetricValue"] {{
        color: {COLOUR_PRIMARY_DARK};
    }}
    </style>
    """, unsafe_allow_html=True)