"""
app.py — vertical sidebar navigation using pure HTML/CSS injected into
st.sidebar. Clicks are captured via st.query_params (no HTML rendering
in button labels). Icons are Unicode symbols styled black, yellow on hover.
"""
 
import logging
import os
import streamlit as st
 
from utils.helpers import get_user_id
from ui.tab_upload    import render_upload_tab
from ui.tab_chat      import render_chat_tab
from ui.tab_dashboard import render_dashboard_tab
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
 
st.set_page_config(
    page_title="ChemSpecAI",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Nav config: (label, key, unicode icon) ───────────────────
_NAV_MAIN = [
    ("Upload",    "upload",    "⬆"),
    ("Chat",      "chat",      "💬"),
    ("Dashboard", "dashboard", "📊"),
]
_NAV_BOTTOM = [
    ("Documents", "docs",  "📄"),
    ("Hub",       "hub",   "🛍"),
    ("Users",     "users", "👤"),
]
 
 
def _init_session() -> None:
    defaults = {
        "azure_bootstrapped": False,
        "dashboard_stats":    None,
        "active_page":        "upload",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    get_user_id(st.session_state)
 
 
def _render_sidebar() -> str:
    active = st.session_state.get("active_page", "upload")
 
    with st.sidebar:
        # ── CSS ───────────────────────────────────────────────
        st.markdown("""
        <style>
        [data-testid="stSidebarHeader"]         { display:none !important; }
        [data-testid="stSidebarCollapseButton"] { display:none !important; }
 
        /* Sidebar: narrow, white */
        [data-testid="stSidebar"] {
            min-width: 72px !important;
            max-width: 72px !important;
            background: #FFFFFF !important;
            border-right: 1px solid #E5E7EB !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 0 !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding: 0 !important;
        }
 
        /* Nav buttons: no box, just icon — circle on hover */
        [data-testid="stSidebar"] .stButton > button {
            background: transparent !important;
            color: #111827 !important;
            border: none !important;
            border-radius: 50% !important;
            width: 44px !important;
            height: 44px !important;
            font-size: 20px !important;
            padding: 0 !important;
            margin: 6px auto !important;
            box-shadow: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: background 0.15s !important;
        }
        /* Hover: yellow circle */
        [data-testid="stSidebar"] .stButton > button:hover {
            background: #FEF08A !important;
            color: #111827 !important;
        }
        /* Remove focus ring */
        [data-testid="stSidebar"] .stButton > button:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        /* Button paragraph (label) */
        [data-testid="stSidebar"] .stButton > button p {
            font-size: 20px !important;
            margin: 0 !important;
            line-height: 1 !important;
            color: #111827 !important;
            filter: grayscale(1) brightness(0);
        }
        </style>
        """, unsafe_allow_html=True)
 
        # ── Logo ──────────────────────────────────────────────
        logo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logo.jpg"
        )
        if os.path.exists(logo_path):
            st.image(logo_path, width=52)
        else:
            st.markdown(
                '<div style="width:52px;height:52px;background:#F3F4F6;'
                'border:2px dashed #D1D5DB;border-radius:6px;'
                'margin:10px auto 0 auto;"></div>',
                unsafe_allow_html=True,
            )
 
        # Separator after logo
        st.markdown(
            '<hr style="margin:8px 0;border:none;border-top:1px solid #E5E7EB;">',
            unsafe_allow_html=True,
        )
 
        # ── Main nav ──────────────────────────────────────────
        for label, key, icon in _NAV_MAIN:
            is_active = (active == key)
 
            # Active item: yellow circle
            if is_active:
                st.markdown(
                    f'<div style="'
                    f'width:44px;height:44px;'
                    f'background:#FEF08A;'
                    f'border-radius:50%;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:20px;color:#111827;'
                    f'margin:6px auto;">'
                    f'{icon}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(icon, key=f"nav_{key}", help=label,
                             use_container_width=True):
                    st.session_state["active_page"] = key
                    st.rerun()
 
        # Separator between groups
        st.markdown(
            '<hr style="margin:4px 0;border:none;border-top:1px solid #D1D5DB;">',
            unsafe_allow_html=True,
        )
 
        # ── Bottom nav ────────────────────────────────────────
        for label, key, icon in _NAV_BOTTOM:
            is_active = (active == key)
            if is_active:
                st.markdown(
                    f'<div style="'
                    f'width:44px;height:44px;'
                    f'background:#FEF08A;'
                    f'border-radius:50%;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:20px;color:#111827;'
                    f'margin:6px auto;">'
                    f'{icon}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(icon, key=f"nav_{key}", help=label,
                             use_container_width=True):
                    st.session_state["active_page"] = key
                    st.rerun()
 
    return st.session_state.get("active_page", "upload")
 
 
def _yellow_boxes() -> str:
    box = (
        '<span style="display:inline-block;width:10px;height:10px;'
        'background:#F5C518;border-radius:2px;margin-left:5px;'
        'vertical-align:middle;"></span>'
    )
    return box * 3
 
 
def _render_header() -> None:
    title_col, user_col = st.columns([5, 1])
    with title_col:
        st.markdown(
            f'<h1 style="margin-bottom:2px;">'
            f'ChemSpecAI'
            f'{_yellow_boxes()}'
            f'</h1>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Dynamic SDS/TDS-Aware RAG · Azure OpenAI · "
            "Azure AI Search · Azure Blob Storage"
        )
    with user_col:
        user_id = st.session_state.get("user_id", "")
        st.caption("Session ID")
        st.code(user_id[:12] + "...", language=None)
 
 
def _render_placeholder(title: str) -> None:
    st.markdown(
        f'<h2>{title}{_yellow_boxes()}</h2>',
        unsafe_allow_html=True,
    )
    st.info(
        f"The **{title}** section is not yet implemented. "
        "Coming in a future phase.", icon="🚧"
    )
 
 
def main() -> None:
    _init_session()
    active = _render_sidebar()
    _render_header()
    st.divider()
 
    if active == "upload":
        render_upload_tab()
    elif active == "chat":
        render_chat_tab()
    elif active == "dashboard":
        render_dashboard_tab()
    elif active == "docs":
        _render_placeholder("Documents")
    elif active == "hub":
        _render_placeholder("Invoice Validation Hub")
    elif active == "users":
        _render_placeholder("User Management")
    else:
        render_upload_tab()
 
 
if __name__ == "__main__":
    main()

