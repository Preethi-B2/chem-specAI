"""
ui/tab_dashboard.py
────────────────────
Tab 3 — Retrieval Quality & Analytics Dashboard
 
Replaces the previous document-stats-only dashboard with a
retrieval quality view centred on similarity scores, plus the
original document index stats below.
 
Layout (top → bottom):
  1. Session retrieval quality KPIs  — avg/best/worst score, queries run
  2. Score quality gauge             — colour-coded score band indicator
  3. Score distribution bar chart    — how scores are spread across buckets
  4. Per-query breakdown table       — every query with its score + doc type
  5. Divider
  6. Document index stats            — total chunks, SDS/TDS split, files
  7. SDS vs TDS chart + uploads table
 
Data sources:
  - Similarity metrics  → st.session_state["query_log"] (populated by tab_chat.py)
  - Document index stats→ Azure AI Search via get_dashboard_stats()
 
tab_chat.py must append to st.session_state["query_log"] after every
successful run_query() call. Each entry is a dict:
  {
    "question":    str,
    "doc_type":    "sds" | "tds",
    "chunks":      list[dict],   # each has "score" key
    "iterations":  int,
    "tool_called": bool,
  }
"""
 
from __future__ import annotations
 
import logging
from datetime import datetime
 
import pandas as pd
import streamlit as st
 
from services.search_service import get_dashboard_stats
from utils.helpers import get_user_id
 
logger = logging.getLogger(__name__)
 
# ── Score band thresholds ──────────────────────────────────────
# Calibrated for text-embedding-3-small (1536 dims).
# Scores from small are naturally compressed vs large (3072 dims),
# so thresholds are shifted ~0.05 lower than large equivalents.
#
# If you upgrade to text-embedding-3-large, use:
#   Excellent ≥ 0.90, Good ≥ 0.80, Fair ≥ 0.70
_SCORE_EXCELLENT = 0.85   # ≥ this → green
_SCORE_GOOD      = 0.75   # ≥ this → blue
_SCORE_FAIR      = 0.65   # ≥ this → orange
                           # < 0.65  → red
 
def _yellow_boxes() -> str:
    box = (
        '<span style="display:inline-block;width:10px;height:10px;'
        'background:#F5C518;border-radius:2px;margin-left:5px;'
        'vertical-align:middle;"></span>'
    )
    return box * 3

def render_dashboard_tab() -> None:
    """
    Render the full Tab 3 — Retrieval Quality & Analytics Dashboard.
    Called directly from app.py inside the st.tabs() block.
    """
    st.header("Retrieval Quality Dashboard")
    st.caption(
        "Similarity scores measure how closely retrieved chunks matched your questions. "
        "Scores range from 0 (no match) to 1 (perfect match). "
        "Thresholds are calibrated for text-embedding-3-small."
    )
 
    # ── Model info banner ─────────────────────────────────────
 
 
    # ── Refresh button ────────────────────────────────────────
    _, col_refresh = st.columns([5, 1])
    with col_refresh:
        refresh = st.button("🔄 Refresh", use_container_width=True)
 
    # ── Section 1: Session retrieval quality ──────────────────
    query_log: list[dict] = st.session_state.get("query_log", [])
    _render_retrieval_quality_section(query_log)
 
    st.divider()
 
    # ── Section 2: Document index stats (from Azure Search) ───
    st.subheader(" Document Index Stats")
    st.caption("Live counts from Azure AI Search.")
 
    if refresh or "dashboard_stats" not in st.session_state:
        with st.spinner("📡 Fetching from Azure AI Search..."):
            try:
                stats = get_dashboard_stats()
                st.session_state["dashboard_stats"] = stats
            except Exception as e:
                st.error(
                    f"❌ Failed to fetch index stats: {e}\n\n"
                    "Check your Azure AI Search configuration."
                )
                logger.exception("Dashboard stats fetch failed.")
                return
 
    stats = st.session_state.get("dashboard_stats", {})
 
    if not stats:
        _render_empty_index_state()
    else:
        _render_index_kpi_metrics(stats)
        st.divider()
        left_col, right_col = st.columns([1, 2])
        with left_col:
            _render_distribution_chart(stats)
        with right_col:
            _render_recent_uploads_table(stats)
 
 
# ═══════════════════════════════════════════════════════════════
# Section 1 — Retrieval quality helpers
# ═══════════════════════════════════════════════════════════════
 
def _render_retrieval_quality_section(query_log: list[dict]) -> None:
    """
    Render all retrieval quality KPIs from the current session's query log.
    Shows an empty-state prompt if no queries have been run yet.
    """
    st.subheader(" Retrieval Quality — This Session")
 
    if not query_log:
        st.info(
            "💬 No queries run yet this session. "
            "Ask a question in the **Chat** tab and come back here to see your retrieval scores."
        )
        return
 
    # ── Flatten all chunk scores across all queries ────────────
    all_scores: list[float] = []
    for entry in query_log:
        for chunk in entry.get("chunks", []):
            score = chunk.get("score")
            if isinstance(score, (int, float)):
                all_scores.append(float(score))
 
    queries_run   = len(query_log)
    chunks_total  = len(all_scores)
    avg_score     = sum(all_scores) / len(all_scores) if all_scores else 0.0
    best_score    = max(all_scores) if all_scores else 0.0
    worst_score   = min(all_scores) if all_scores else 0.0
 
    # ── KPI metric row ─────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
 
    with c1:
        st.metric(
            label="💬 Queries Run",
            value=queries_run,
            help="Total questions asked this session.",
        )
    with c2:
        st.metric(
            label="📎 Chunks Retrieved",
            value=chunks_total,
            help="Total document chunks retrieved across all queries.",
        )
    with c3:
        quality_label = _score_label(avg_score)
        st.metric(
            label="📊 Avg Similarity Score",
            value=f"{avg_score:.3f}",
            help=f"Mean similarity score across all retrieved chunks. Quality: {quality_label}",
        )
    with c4:
        st.metric(
            label="🏆 Best Score",
            value=f"{best_score:.3f}",
            help="Highest similarity score seen this session — the most confident retrieval.",
        )
    with c5:
        st.metric(
            label="⚠️ Worst Score",
            value=f"{worst_score:.3f}",
            help="Lowest similarity score seen — the least confident retrieval.",
        )
 
    st.markdown("")
 
    # ── Quality gauge ──────────────────────────────────────────
    _render_quality_gauge(avg_score)
 
    st.markdown("")
 
    # ── Score distribution chart ───────────────────────────────
    if all_scores:
        _render_score_distribution(all_scores)
 
    st.markdown("")
 
    # ── Per-query breakdown table ──────────────────────────────
    _render_query_breakdown_table(query_log)
 
 
def _score_label(score: float) -> str:
    """Return a human-readable quality label for a given score."""
    if score >= _SCORE_EXCELLENT:
        return "Excellent"
    elif score >= _SCORE_GOOD:
        return "Good"
    elif score >= _SCORE_FAIR:
        return "Fair"
    else:
        return "Poor"
 
 
def _score_colour(score: float) -> str:
    """Return a hex colour matching the quality band."""
    if score >= _SCORE_EXCELLENT:
        return "#2E7D32"   # green
    elif score >= _SCORE_GOOD:
        return "#1565C0"   # blue
    elif score >= _SCORE_FAIR:
        return "#E65100"   # orange
    else:
        return "#B71C1C"   # red
 
 
def _render_quality_gauge(avg_score: float) -> None:
    """
    Render a colour-coded quality gauge as an HTML banner.
    Shows the average score and its quality band with an explanation.
    """
    label  = _score_label(avg_score)
    colour = _score_colour(avg_score)
 
    band_explanations = {
        "Excellent": "≥ 0.85 — Strong retrieval for text-embedding-3-small. "
                     "The chunks are highly relevant and answers are very likely accurate.",
        "Good":      "0.75–0.84 — Solid retrieval. Most chunks are relevant. "
                     "Occasional noise is normal and GPT-4o mini handles it well.",
        "Fair":      "0.65–0.74 — Related content found but not tightly matched. "
                     "Try rephrasing with specific chemistry terminology (CAS numbers, "
                     "section names, formal parameter names).",
        "Poor":      "< 0.65 — Low confidence retrieval for this model. "
                     "Rephrase your query or verify the correct document is uploaded.",
    }
 
    explanation = band_explanations.get(label, "")
 
    st.markdown(
        f"""
        <div style="
            background: {colour}18;
            border-left: 5px solid {colour};
            border-radius: 8px;
            padding: 14px 20px;
            margin-bottom: 4px;
        ">
            <span style="font-size:1.5rem; font-weight:700; color:{colour};">
                {label} — {avg_score:.3f}
            </span>
            <br/>
            <span style="font-size:0.88rem; color:#444; margin-top:4px; display:block;">
                {explanation}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
def _render_score_distribution(scores: list[float]) -> None:
    """
    Render a bar chart showing how scores are spread across quality buckets.
    Buckets: <0.70, 0.70–0.79, 0.80–0.89, ≥0.90
    """
    st.markdown("**Score Distribution**")
    st.caption("How your retrieved chunks are spread across similarity score bands.")
 
    buckets = {
        "Poor  (<0.65)":      0,
        "Fair  (0.65–0.74)":  0,
        "Good  (0.75–0.84)":  0,
        "Excellent  (≥0.85)": 0,
    }
 
    for s in scores:
        if s >= 0.85:
            buckets["Excellent  (≥0.85)"] += 1
        elif s >= 0.75:
            buckets["Good  (0.75–0.84)"] += 1
        elif s >= 0.65:
            buckets["Fair  (0.65–0.74)"] += 1
        else:
            buckets["Poor  (<0.65)"] += 1
 
    df = pd.DataFrame({
        "Band":  list(buckets.keys()),
        "Chunks": list(buckets.values()),
    }).set_index("Band")
 
    st.bar_chart(
        df,
        color=["#1565C0"],
        height=220,
    )
 
    # Caption with bucket percentages
    total = len(scores)
    parts = [
        f"**{label.split('(')[0].strip()}**: {count} ({count/total*100:.0f}%)"
        for label, count in buckets.items()
        if count > 0
    ]
    st.caption("  |  ".join(parts))
 
def _render_query_breakdown_table(query_log: list[dict]) -> None:
    """
    Render a table showing every query with its avg score, doc type,
    chunks retrieved, and number of iterations.
    Most recent query shown first.
    """
    st.markdown("**Per-Query Breakdown**")
 
    rows = []
    for i, entry in enumerate(reversed(query_log), start=1):
        chunks = entry.get("chunks", [])
        scores = [
            float(c["score"])
            for c in chunks
            if isinstance(c.get("score"), (int, float))
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        best = max(scores) if scores else 0.0
        doc_type = entry.get("doc_type", "").upper()
        badge = "🟠 SDS" if doc_type == "SDS" else "🔵 TDS"
        label = _score_label(avg)
 
        rows.append({
            "#":            len(query_log) - i + 1,
            "Question":     entry.get("question", "")[:80],
            "Type":         badge,
            "Avg Score":    round(avg, 3),
            "Best Score":   round(best, 3),
            "Quality":      label,
            "Chunks":       len(chunks),
            "Iterations":   entry.get("iterations", 1),
        })
 
    df = pd.DataFrame(rows)
 
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#":          st.column_config.NumberColumn("#",           width="small"),
            "Question":   st.column_config.TextColumn("Question",      width="large"),
            "Type":       st.column_config.TextColumn("Doc Type",      width="small"),
            "Avg Score":  st.column_config.NumberColumn("Avg Score",   format="%.3f", width="small"),
            "Best Score": st.column_config.NumberColumn("Best Score",  format="%.3f", width="small"),
            "Quality":    st.column_config.TextColumn("Quality",       width="small"),
            "Chunks":     st.column_config.NumberColumn("Chunks",      width="small"),
            "Iterations": st.column_config.NumberColumn("Iterations",  width="small"),
        },
    )
    st.caption(f"Showing all {len(rows)} queries — newest first.")
 
 
# ═══════════════════════════════════════════════════════════════
# Section 2 — Document index stats helpers (unchanged logic)
# ═══════════════════════════════════════════════════════════════
 
def _render_index_kpi_metrics(stats: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Chunks",  stats.get("total_chunks", 0),
                  help="Total indexed text chunks across all documents.")
    with col2:
        st.metric("🟠 SDS Chunks",    stats.get("sds_count", 0),
                  help="Chunks from Safety Data Sheet documents.")
    with col3:
        st.metric("🔵 TDS Chunks",    stats.get("tds_count", 0),
                  help="Chunks from Technical Data Sheet documents.")
    with col4:
        st.metric("📄 Unique Files",  stats.get("unique_files", 0),
                  help="Number of distinct documents uploaded.")
 
 
def _render_distribution_chart(stats: dict) -> None:
    st.subheader("📈 SDS vs TDS Distribution")
    sds_count = stats.get("sds_count", 0)
    tds_count = stats.get("tds_count", 0)
    if sds_count == 0 and tds_count == 0:
        st.info("No chunks indexed yet.")
        return
    df = pd.DataFrame(
        {"Document Type": ["SDS", "TDS"], "Chunk Count": [sds_count, tds_count]}
    ).set_index("Document Type")
    st.bar_chart(df, color=["#FF6B35"], height=260)
    total = sds_count + tds_count
    if total > 0:
        st.caption(
            f"🟠 SDS: {sds_count/total*100:.1f}%  |  🔵 TDS: {tds_count/total*100:.1f}%"
        )
 
 
def _render_recent_uploads_table(stats: dict) -> None:
    st.subheader("🕐 Recent Uploads")
    recent: list[dict] = stats.get("recent_uploads", [])
    if not recent:
        st.info("No documents uploaded yet.")
        return
    rows = [
        {
            "File": doc.get("source", "Unknown"),
            "Type": "🟠 SDS" if doc.get("type", "").upper() == "SDS" else "🔵 TDS",
        }
        for doc in recent
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "File": st.column_config.TextColumn("📄 File Name", width="large"),
            "Type": st.column_config.TextColumn("Type",         width="small"),
        },
    )
    st.caption(f"Showing most recent {len(rows)} unique document(s).")
 
 
def _render_empty_index_state() -> None:
    st.markdown(
        """
        <div style="
            text-align:center; padding:3rem 1rem;
            border:2px dashed #ccc; border-radius:12px; color:#888;
        ">
            <h3>📊 No documents indexed yet</h3>
            <p>Upload chemistry documents in the <strong>Upload</strong> tab
               to see index stats here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
def _format_timestamp(iso_timestamp: str) -> str:
    if not iso_timestamp:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%d %b %Y, %H:%M UTC")
    except (ValueError, TypeError):
        return iso_timestamp[:19].replace("T", " ")