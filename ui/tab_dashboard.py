"""
ui/tab_dashboard.py
────────────────────
Tab 3 — Analytics Dashboard
 
Responsibilities (UI only):
  1. Fetch live analytics from Azure AI Search via search_service
  2. Display total chunks, SDS count, TDS count, unique files
  3. Render recent uploads table
  4. Render SDS vs TDS distribution chart
  5. Provide manual refresh button
 
All data is fetched dynamically — no static or preloaded JSON.
All queries are scoped to the current user_id.
"""
 
from __future__ import annotations
 
import logging
from datetime import datetime
 
import streamlit as st
 
from services.search_service import get_dashboard_stats
from utils.helpers import get_user_id
 
logger = logging.getLogger(__name__)
 
 
def render_dashboard_tab() -> None:
    """
    Render the full Tab 3 — Analytics Dashboard.
    Called directly from app.py inside the st.tabs() block.
    """
    user_id = get_user_id(st.session_state)
 
    # ── Header ────────────────────────────────────────────────
    st.header("📊 Document Analytics Dashboard")
    st.caption("Live analytics fetched from Azure AI Search. Scoped to your session.")
 
    # ── Refresh control ───────────────────────────────────────
    col_title, col_refresh = st.columns([4, 1])
    with col_refresh:
        refresh = st.button("🔄 Refresh", use_container_width=True)
 
    # ── Fetch data ────────────────────────────────────────────
    # Cache key in session_state to avoid re-fetching on every Streamlit rerun
    # Cleared when user presses Refresh
    if refresh or "dashboard_stats" not in st.session_state:
        with st.spinner("📡 Fetching analytics from Azure AI Search..."):
            try:
                stats = get_dashboard_stats()
                st.session_state["dashboard_stats"] = stats
            except Exception as e:
                st.error(
                    f"❌ Failed to fetch dashboard data: {e}\n\n"
                    "Ensure Azure AI Search is configured correctly."
                )
                logger.exception("Dashboard stats fetch failed.")
                return
 
    stats = st.session_state.get("dashboard_stats", {})
 
    if not stats:
        _render_empty_dashboard()
        return
 
    # ── Top-level KPI metrics ─────────────────────────────────
    _render_kpi_metrics(stats)
 
    st.divider()
 
    # ── SDS vs TDS chart + recent uploads table ───────────────
    left_col, right_col = st.columns([1, 2])
 
    with left_col:
        _render_distribution_chart(stats)
 
    with right_col:
        _render_recent_uploads_table(stats)
 
 
def _render_kpi_metrics(stats: dict) -> None:
    """
    Render the 4 top-level KPI metric cards.
 
    Args:
        stats: Dict returned by get_dashboard_stats().
    """
    col1, col2, col3, col4 = st.columns(4)
 
    with col1:
        st.metric(
            label="📦 Total Chunks",
            value=stats.get("total_chunks", 0),
            help="Total indexed text chunks across all your documents.",
        )
 
    with col2:
        st.metric(
            label="🟠 SDS Chunks",
            value=stats.get("sds_count", 0),
            help="Chunks from Safety Data Sheet documents.",
        )
 
    with col3:
        st.metric(
            label="🔵 TDS Chunks",
            value=stats.get("tds_count", 0),
            help="Chunks from Technical Data Sheet documents.",
        )
 
    with col4:
        st.metric(
            label="📄 Unique Files",
            value=stats.get("unique_files", 0),
            help="Number of distinct documents uploaded.",
        )
 
 
def _render_distribution_chart(stats: dict) -> None:
    """
    Render an SDS vs TDS donut-style bar chart using st.bar_chart.
 
    Args:
        stats: Dict returned by get_dashboard_stats().
    """
    st.subheader("📈 SDS vs TDS Distribution")
 
    sds_count = stats.get("sds_count", 0)
    tds_count = stats.get("tds_count", 0)
 
    if sds_count == 0 and tds_count == 0:
        st.info("No chunks indexed yet.")
        return
 
    chart_data = {
        "Document Type": ["SDS", "TDS"],
        "Chunk Count":   [sds_count, tds_count],
    }
 
    import pandas as pd
    df = pd.DataFrame(chart_data).set_index("Document Type")
    st.bar_chart(
        df,
        color=["#FF6B35"],
        height=260,
    )
 
    # Percentage breakdown
    total = sds_count + tds_count
    if total > 0:
        sds_pct = sds_count / total * 100
        tds_pct = tds_count / total * 100
        st.caption(f"🟠 SDS: {sds_pct:.1f}%  |  🔵 TDS: {tds_pct:.1f}%")
 
 
def _render_recent_uploads_table(stats: dict) -> None:
    """
    Render the recent uploads table showing last 10 distinct documents.
 
    Args:
        stats: Dict returned by get_dashboard_stats(), includes 'recent_uploads' list.
    """
    st.subheader("🕐 Recent Uploads")
 
    recent: list[dict] = stats.get("recent_uploads", [])
 
    if not recent:
        st.info("No documents uploaded yet.")
        return
 
    import pandas as pd
 
    rows = []
    for doc in recent:
        doc_type = doc.get("type", "").upper()
        badge = "🟠 SDS" if doc_type == "SDS" else "🔵 TDS"
 
        rows.append({
            "File":  doc.get("source", "Unknown"),
            "Type":  badge,
        })
 
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "File":     st.column_config.TextColumn("📄 File Name", width="large"),
            "Type":     st.column_config.TextColumn("Type",         width="small"),
            "Uploaded": st.column_config.TextColumn("🕐 Uploaded",  width="medium"),
        },
    )
 
    st.caption(f"Showing most recent {len(rows)} unique document(s).")
 
 
def _render_empty_dashboard() -> None:
    """Render the empty state when no documents have been uploaded yet."""
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 3rem 1rem;
            border: 2px dashed #ccc;
            border-radius: 12px;
            color: #888;
        ">
            <h3>📊 No data yet</h3>
            <p>Upload chemistry documents in the <strong>Upload</strong> tab to see analytics here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
def _format_timestamp(iso_timestamp: str) -> str:
    """
    Format an ISO 8601 timestamp string to a human-readable form.
 
    Args:
        iso_timestamp: e.g. "2026-03-04T10:30:00+00:00"
 
    Returns:
        e.g. "04 Mar 2026, 10:30 UTC"
    """
    if not iso_timestamp:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%d %b %Y, %H:%M UTC")
    except (ValueError, TypeError):
        return iso_timestamp[:19].replace("T", " ")
 