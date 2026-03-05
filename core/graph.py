"""
core/graph.py
──────────────
Builds the LangGraph RAG pipeline and exposes run_query() —
the same function signature the UI already calls in tab_chat.py.
 
Graph structure:
                    START
                      │
               classify_node        ← 'sds' or 'tds'?
                      │
                 agent_node         ← LLM with tools bound
                      │
            should_continue()       ← conditional edge
           ┌──────────┴──────────┐
         'tools'               'end'
           │                     │
        tool_node         collect_chunks_node
      (ToolNode)                  │
           │                     END
           └──► agent_node
 
The graph is compiled once at module load and reused for every
user question — no rebuild cost per request.
 
Public interface (unchanged from old query_engine.py):
    from core.graph import run_query, QueryResult
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass, field
 
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
 
from core.state import GraphState
from core.tools import TOOLS
from core.nodes import (
    classify_node,
    agent_node,
    collect_chunks_node,
    should_continue,
)
from utils.prompt_loader import load_prompt
 
logger = logging.getLogger(__name__)
 
 
# ── QueryResult ───────────────────────────────────────────────
# Identical dataclass to the old query_engine.py — UI unchanged.
 
@dataclass
class QueryResult:
    """
    Final result returned to ui/tab_chat.py after graph execution.
 
    Fields match the old QueryResult exactly so tab_chat.py needs
    zero changes beyond the import line.
    """
    answer:           str
    doc_type:         str
    chunks_retrieved: list[dict]
    tool_was_called:  bool
    iterations:       int
    error:            str | None = None
 
 
# ── Graph builder ─────────────────────────────────────────────
 
def _build_graph():
    """
    Construct and compile the LangGraph StateGraph.
 
    Called once at module load. Returns a compiled graph object
    that can be invoked repeatedly without rebuilding.
    """
    g = StateGraph(GraphState)
 
    # ── Register nodes ────────────────────────────────────────
    g.add_node("classify",       classify_node)
    g.add_node("agent",          agent_node)
    g.add_node("tools",          ToolNode(TOOLS))   # prebuilt — executes @tool functions
    g.add_node("collect_chunks", collect_chunks_node)
 
    # ── Entry point ───────────────────────────────────────────
    g.set_entry_point("classify")
 
    # ── Fixed edges ───────────────────────────────────────────
    g.add_edge("classify",       "agent")           # classify → always goes to agent
    g.add_edge("tools",          "agent")           # after tool execution → back to agent
    g.add_edge("collect_chunks", END)               # after chunks collected → done
 
    # ── Conditional edge ──────────────────────────────────────
    # After agent_node, should_continue() decides the next node.
    g.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",           # LLM called a tool → execute it
            "end":   "collect_chunks",  # LLM gave an answer → collect + finish
        },
    )
 
    return g.compile()
 
 
# Compiled once at import time — reused for every question
_graph = _build_graph()
 
 
# ── Public entry point ────────────────────────────────────────
 
def run_query(
    user_question:             str,
    user_id:                   str,
    memory_messages:           list[dict],
    conversation_history_text: str,
) -> QueryResult:
    """
    Run the full LangGraph RAG pipeline for a single user question.
 
    This function has the IDENTICAL signature to the old run_query()
    in core/query_engine.py — tab_chat.py only needs its import line changed.
 
    Flow:
      1. Convert memory dicts → LangChain message objects
      2. Build initial GraphState with system prompt + memory + question
      3. Invoke the compiled graph
      4. Extract final AIMessage and chunk metadata
      5. Return QueryResult to the UI
 
    Args:
        user_question:             The user's current question string.
        user_id:                   Session user ID (kept for signature compat).
        memory_messages:           Prior turns as list of {"role","content"} dicts.
        conversation_history_text: Plain-text memory for prompt placeholders.
 
    Returns:
        QueryResult with answer, doc_type, chunks, metadata, and optional error.
    """
    logger.info(f"[run_query] Starting graph for: '{user_question[:60]}...'")
 
    # ── Step 1: Convert memory dicts → LangChain message objects ──
    lc_memory: list = []
    for m in memory_messages:
        if m["role"] == "user":
            lc_memory.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_memory.append(AIMessage(content=m["content"]))
 
    # ── Step 2: Build initial state ───────────────────────────────
    system_prompt = load_prompt("system_prompt.md")
 
    initial_state: GraphState = {
        "messages": [
            SystemMessage(content=system_prompt),
            *lc_memory,
            HumanMessage(content=user_question),
        ],
        "question":         user_question,
        "doc_type":         "sds",      # overwritten by classify_node
        "chunks_retrieved": [],
        "tool_was_called":  False,
        "iterations":       0,
        "error":            None,
    }
 
    # ── Step 3: Run the graph ─────────────────────────────────────
    try:
        final_state: GraphState = _graph.invoke(initial_state)
 
    except Exception as e:
        logger.exception("[run_query] Graph execution failed.")
        return QueryResult(
            answer=(
                "An unexpected error occurred while processing your question. "
                "Please try again."
            ),
            doc_type="sds",
            chunks_retrieved=[],
            tool_was_called=False,
            iterations=0,
            error=str(e),
        )
 
    # ── Step 4: Extract final answer from last AIMessage ──────────
    ai_messages = [
        m for m in final_state["messages"]
        if isinstance(m, AIMessage) and m.content
    ]
 
    if ai_messages:
        final_answer = ai_messages[-1].content
    else:
        logger.warning("[run_query] No AIMessage with content found in final state.")
        final_answer = (
            "I was unable to generate an answer. Please try rephrasing your question."
        )
 
    logger.info(
        f"[run_query] Completed in {final_state.get('iterations', 0)} iteration(s). "
        f"Chunks: {len(final_state.get('chunks_retrieved', []))}. "
        f"doc_type: {final_state.get('doc_type', 'unknown')}."
    )
 
    # ── Step 5: Return QueryResult to UI ─────────────────────────
    return QueryResult(
        answer=final_answer,
        doc_type=final_state.get("doc_type", "sds"),
        chunks_retrieved=final_state.get("chunks_retrieved", []),
        tool_was_called=final_state.get("tool_was_called", False),
        iterations=final_state.get("iterations", 0),
        error=final_state.get("error"),
    )
 
