"""
core/memory_manager.py
───────────────────────
Short-term conversational memory backed by Streamlit session_state.
 
Characteristics:
  - Stores last N turns (configurable via MAX_MEMORY_TURNS in settings)
  - Lives entirely in st.session_state — no database, no Azure calls
  - Cleared automatically when the browser session ends
  - Never embedded or indexed
  - Used only to provide context to the LLM — never replaces retrieval
 
Memory Turn Schema:
    {
        "role":    "user" | "assistant",
        "content": str,
    }
 
This matches the OpenAI messages[] format directly, so memory turns
can be injected into chat_completion() messages without transformation.
"""
 
from __future__ import annotations
 
import logging
 
from config.settings import MAX_MEMORY_TURNS
 
logger = logging.getLogger(__name__)
 
# Key used to store the memory list in st.session_state
_MEMORY_KEY = "conversation_memory"
 
 
# ── Initialization ────────────────────────────────────────────
 
def init_memory(session_state: dict) -> None:
    """
    Ensure the memory list exists in session state.
    Safe to call on every app render — idempotent.
 
    Args:
        session_state: Streamlit's st.session_state dict.
    """
    if _MEMORY_KEY not in session_state:
        session_state[_MEMORY_KEY] = []
        logger.debug("Memory initialized in session state.")
 
 
# ── Read ──────────────────────────────────────────────────────
 
def get_memory(session_state: dict) -> list[dict[str, str]]:
    """
    Retrieve the current conversation memory as a list of message dicts.
 
    Returns the last MAX_MEMORY_TURNS turns only. Oldest turns are
    dropped automatically when the window fills up.
 
    Args:
        session_state: Streamlit's st.session_state dict.
 
    Returns:
        List of {"role": ..., "content": ...} dicts in chronological order.
        Ready to be injected directly into OpenAI messages[].
    """
    init_memory(session_state)
    memory: list[dict] = session_state[_MEMORY_KEY]
 
    # Each "turn" = 1 user message + 1 assistant message = 2 entries
    max_entries = MAX_MEMORY_TURNS * 2
    return memory[-max_entries:] if len(memory) > max_entries else list(memory)
 
 
def format_memory_as_text(session_state: dict) -> str:
    """
    Format memory as a plain text string for injection into prompt templates
    that use {conversation_history} placeholders (e.g. query_classifier.md).
 
    Args:
        session_state: Streamlit's st.session_state dict.
 
    Returns:
        Multi-line string like:
          User: What is the flash point?
          Assistant: The flash point is 23°C according to section 9.
        Returns empty string if no memory exists yet.
    """
    memory = get_memory(session_state)
    if not memory:
        return ""
 
    lines: list[str] = []
    for msg in memory:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
 
    return "\n".join(lines)
 
 
# ── Write ─────────────────────────────────────────────────────
 
def add_turn(
    session_state: dict,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    Append a completed conversation turn (user + assistant) to memory.
    Call this AFTER the assistant has generated its final answer.
 
    Automatically trims memory to the last MAX_MEMORY_TURNS turns
    to prevent unbounded growth within a session.
 
    Args:
        session_state:      Streamlit's st.session_state dict.
        user_message:       The user's question text.
        assistant_message:  The assistant's answer text.
    """
    init_memory(session_state)
 
    session_state[_MEMORY_KEY].append({"role": "user",      "content": user_message})
    session_state[_MEMORY_KEY].append({"role": "assistant", "content": assistant_message})
 
    # Trim: keep only last N turns (N turns = N*2 message entries)
    max_entries = MAX_MEMORY_TURNS * 2
    if len(session_state[_MEMORY_KEY]) > max_entries:
        session_state[_MEMORY_KEY] = session_state[_MEMORY_KEY][-max_entries:]
 
    logger.debug(
        f"Memory updated. "
        f"Turns stored: {len(session_state[_MEMORY_KEY]) // 2}/{MAX_MEMORY_TURNS}"
    )
 
 
# ── Clear ─────────────────────────────────────────────────────
 
def clear_memory(session_state: dict) -> None:
    """
    Wipe all conversation memory for the current session.
    Exposed to the UI so users can reset the chat context.
 
    Args:
        session_state: Streamlit's st.session_state dict.
    """
    session_state[_MEMORY_KEY] = []
    logger.info("Conversation memory cleared.")
 
 
# ── Introspection ─────────────────────────────────────────────
 
def get_turn_count(session_state: dict) -> int:
    """
    Return the number of completed conversation turns stored.
 
    Args:
        session_state: Streamlit's st.session_state dict.
 
    Returns:
        Integer count of turns (each turn = 1 user + 1 assistant message).
    """
    init_memory(session_state)
    return len(session_state[_MEMORY_KEY]) // 2
 
