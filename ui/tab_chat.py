"""
ui/tab_chat.py
───────────────
Tab 2 — Chat
 
Responsibilities (UI only):
  1. Render chat message history from session_state
  2. Accept user input via st.chat_input
  3. Call core/memory_manager.py to read/write memory
  4. Call core/query_engine.py → run_query() for each question
  5. Display answer + retrieval metadata (source chunks, doc_type badge)
  6. Provide memory reset button
 
Rule: This tab contains zero business logic.
All RAG orchestration happens in core/query_engine.py.
"""
 
from __future__ import annotations
 
import logging
 
import streamlit as st
 
from core.memory_manager import (
    get_memory,
    format_memory_as_text,
    add_turn,
    clear_memory,
    get_turn_count,
)
from core.graph import run_query, QueryResult
from utils.helpers import get_user_id
 
logger = logging.getLogger(__name__)
 
# Session state keys
_CHAT_HISTORY_KEY = "chat_display_history"
 
 
def render_chat_tab() -> None:
    """
    Render the full Tab 2 — Chat interface.
    Called directly from app.py inside the st.tabs() block.
    """
    user_id = get_user_id(st.session_state)
 
    # ── Header ────────────────────────────────────────────────
    st.header("Chemistry Document Chat")


    st.caption(
        "Ask questions about your uploaded SDS or TDS documents. "
        "The system retrieves relevant chunks and generates grounded answers."
    )
 
    # ── Memory status + controls ──────────────────────────────
    _render_memory_controls()
 
    st.divider()
 
    # ── Guard: warn if no documents uploaded yet ──────────────
    _render_upload_reminder()
 
    # ── Initialise display history ────────────────────────────
    if _CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[_CHAT_HISTORY_KEY] = []
 
    # ── Render existing conversation ──────────────────────────
    _render_chat_history()
 
    # ── Chat input ────────────────────────────────────────────
    user_input = st.chat_input(
        placeholder="Ask about safety, handling, specifications, hazards...",
    )
 
    if user_input:
        _handle_user_message(user_input, user_id)
 
 
def _handle_user_message(user_question: str, user_id: str) -> None:
    """
    Process a new user message:
      1. Display user bubble immediately
      2. Call query_engine.run_query()
      3. Display assistant response with metadata
      4. Update memory
 
    Args:
        user_question: The text the user submitted.
        user_id:       Current session user ID.
    """
    # ── Display user message immediately ─────────────────────
    with st.chat_message("user"):
        st.markdown(user_question)
 
    # Add to display history
    st.session_state[_CHAT_HISTORY_KEY].append({
        "role":    "user",
        "content": user_question,
    })
 
    # ── Run the agentic RAG query ─────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving and reasoning..."):
            try:
                memory_messages       = get_memory(st.session_state)
                conversation_history  = format_memory_as_text(st.session_state)
 
                result: QueryResult = run_query(
                    user_question=user_question,
                    user_id=user_id,
                    memory_messages=memory_messages,
                    conversation_history_text=conversation_history,
                )
 
                _render_assistant_response(result)
 
                # ── Update memory with this completed turn ────
                add_turn(
                    session_state=st.session_state,
                    user_message=user_question,
                    assistant_message=result.answer,
                )
 
                # ── Log query for dashboard similarity metrics ─
                # query_log is read by tab_dashboard to compute
                # avg/best/worst similarity scores and the
                # per-query breakdown table.
                if "query_log" not in st.session_state:
                    st.session_state["query_log"] = []
                if result.tool_was_called:
                    st.session_state["query_log"].append({
                        "question":   user_question,
                        "doc_type":   result.doc_type,
                        "chunks":     result.chunks_retrieved,
                        "iterations": result.iterations,
                        "tool_called": result.tool_was_called,
                    })
 
                # Add to display history
                st.session_state[_CHAT_HISTORY_KEY].append({
                    "role":    "assistant",
                    "content": result.answer,
                    "metadata": {
                        "doc_type":       result.doc_type,
                        "tool_called":    result.tool_was_called,
                        "chunks":         result.chunks_retrieved,
                        "iterations":     result.iterations,
                    },
                })
 
            except Exception as e:
                error_msg = (
                    f"❌ An error occurred while processing your question: {e}\n\n"
                    "Please check your Azure configuration and try again."
                )
                st.error(error_msg)
                logger.exception(f"Query failed for question: '{user_question[:60]}...'")
 
 
def _render_assistant_response(result: QueryResult) -> None:
    """
    Render the assistant answer bubble with:
      - Answer text (markdown)
      - Document type badge
      - Retrieval metadata expander (source chunks)
      - Warning if no retrieval occurred
 
    Args:
        result: QueryResult returned by query_engine.run_query()
    """
    # ── Answer text ───────────────────────────────────────────
    st.markdown(result.answer)
 
    # ── Metadata row ──────────────────────────────────────────
    meta_col1, meta_col2, meta_col3 = st.columns([1, 1, 2])
 
    with meta_col1:
        if result.doc_type == "sds":
            st.badge("🟠 SDS", color="orange")
        else:
            st.badge("🔵 TDS", color="blue")
 
    with meta_col2:
        if result.tool_was_called:
            st.badge(f"📄 {len(result.chunks_retrieved)} chunk(s) retrieved", color="green")
        else:
            st.badge("⚠️ No retrieval", color="gray")
 
    with meta_col3:
        st.caption(f"Iterations: {result.iterations}")
 
    # ── Warning if LLM answered without retrieval ─────────────
    if not result.tool_was_called:
        st.warning(
            "⚠️ This answer was not grounded in retrieved document chunks. "
            "Please upload relevant documents first.",
            icon="⚠️",
        )
 
    # ── Error flag ────────────────────────────────────────────
    if result.error:
        st.error(f"System note: {result.error}")
 
    # ── Retrieved chunks expander ─────────────────────────────
    if result.chunks_retrieved:
        with st.expander(
            f"📚 View {len(result.chunks_retrieved)} source chunk(s)",
            expanded=False,
        ):
            for i, chunk in enumerate(result.chunks_retrieved, start=1):
                with st.container(border=True):
                    chunk_col1, chunk_col2 = st.columns([3, 1])
                    with chunk_col1:
                        st.markdown(f"**Source:** `{chunk.get('source', 'Unknown')}`")
                        st.markdown(f"**Section:** {chunk.get('section', 'N/A')}")
                    with chunk_col2:
                        score = chunk.get("score", 0)
                        st.metric("Score", f"{score:.3f}")
 
                    st.markdown(f"> {chunk.get('content', '')[:400]}...")
 
 
def _render_chat_history() -> None:
    """
    Re-render all previous messages from the display history.
    Called on every Streamlit rerun to restore the visible conversation.
    """
    for msg in st.session_state.get(_CHAT_HISTORY_KEY, []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
 
            # Re-render metadata for assistant messages
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]
                m1, m2, m3 = st.columns([1, 1, 2])
 
                with m1:
                    if meta["doc_type"] == "sds":
                        st.badge("🟠 SDS", color="orange")
                    else:
                        st.badge("🔵 TDS", color="blue")
 
                with m2:
                    chunks = meta.get("chunks", [])
                    if meta["tool_called"]:
                        st.badge(f"📄 {len(chunks)} chunk(s)", color="green")
                    else:
                        st.badge("⚠️ No retrieval", color="gray")
 
                if chunks:
                    with st.expander(f"📚 {len(chunks)} source chunk(s)", expanded=False):
                        for chunk in chunks:
                            with st.container(border=True):
                                c1, c2 = st.columns([3, 1])
                                with c1:
                                    st.markdown(f"**Source:** `{chunk.get('source', '')}`")
                                    st.markdown(f"**Section:** {chunk.get('section', 'N/A')}")
                                with c2:
                                    st.metric("Score", f"{chunk.get('score', 0):.3f}")
                                st.markdown(f"> {chunk.get('content', '')[:400]}...")
 
 
def _render_memory_controls() -> None:
    """
    Render the memory status bar and clear button in the sidebar area of the tab.
    """
    turn_count = get_turn_count(st.session_state)
 
    mem_col1, mem_col2 = st.columns([3, 1])
 
    with mem_col1:
        if turn_count == 0:
            st.caption(" Memory: No conversation history yet.")
        else:
            from config.settings import MAX_MEMORY_TURNS
            st.caption(
                f" Memory: {turn_count}/{MAX_MEMORY_TURNS} turns stored "
                f"(session only — cleared on page refresh)"
            )
 
    with mem_col2:
        if turn_count > 0:
            if st.button("🗑️ Clear Memory", use_container_width=True):
                clear_memory(st.session_state)
                st.session_state[_CHAT_HISTORY_KEY] = []
                st.rerun()
 
 
def _render_upload_reminder() -> None:
    """
    Show a reminder banner if no documents have been indexed yet.
    Checks session state flag set by tab_upload after successful indexing.
    """
    # We use azure_bootstrapped as a lightweight proxy for "app is connected"
    # A more robust check would query the index for document count
    if not st.session_state.get("azure_bootstrapped"):
        st.info(
            "💡 **Tip:** Go to the **Upload** tab first to index your chemistry documents "
            "before starting a chat.",
            icon="ℹ️",
        )
