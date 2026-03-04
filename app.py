"""
app.py
───────
Streamlit application entry point.
 
Responsibilities (routing only):
  - Page configuration
  - Session state initialisation
  - 3-tab layout rendering
  - Delegates all logic to ui/ modules
 
Run with:
    streamlit run app.py
 
Tabs:
  Tab 1 — 📂 Upload   → ui/tab_upload.py
  Tab 2 — 💬 Chat     → ui/tab_chat.py
  Tab 3 — 📊 Dashboard → ui/tab_dashboard.py
"""
 
import logging
 
import streamlit as st
 
from utils.helpers import get_user_id
from ui.tab_upload    import render_upload_tab
from ui.tab_chat      import render_chat_tab
from ui.tab_dashboard import render_dashboard_tab
 
# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
 
 
# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Chemistry Doc Intelligence",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
 
# ── Session state initialisation ─────────────────────────────
def _init_session() -> None:
    """
    Initialise all required session state keys on first load.
    Safe to call on every rerun — only sets keys that don't exist yet.
    """
    defaults = {
        "azure_bootstrapped":  False,
        "dashboard_stats":     None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
 
    # Ensure user_id is created once and stable for this session
    get_user_id(st.session_state)
 
 
# ── App header ────────────────────────────────────────────────
def _render_header() -> None:
    """Render the top application header and user session info."""
    title_col, user_col = st.columns([5, 1])
 
    with title_col:
        st.title("⚗️ Chemistry Document Intelligence")
        st.caption(
            "Dynamic SDS/TDS-Aware RAG · Azure OpenAI · Azure AI Search · Azure Blob Storage"
        )
 
    with user_col:
        user_id = st.session_state.get("user_id", "")
        st.caption(f"🔑 Session ID")
        st.code(user_id[:12] + "...", language=None)
 
 
# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    _init_session()
    _render_header()
 
    st.divider()
 
    # ── 3-tab layout ──────────────────────────────────────────
    tab_upload, tab_chat, tab_dashboard = st.tabs([
        "📂 Upload",
        "💬 Chat",
        "📊 Dashboard",
    ])
 
    with tab_upload:
        render_upload_tab()
 
    with tab_chat:
        render_chat_tab()
 
    with tab_dashboard:
        render_dashboard_tab()
 
 
if __name__ == "__main__":
    main()
 
