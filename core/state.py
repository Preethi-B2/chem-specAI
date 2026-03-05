"""
core/state.py
──────────────
Defines the single shared state object that flows through every
node in the LangGraph RAG pipeline.
 
LangGraph passes this state into each node function and merges
the node's return dict back into the state automatically.
 
Key design — Annotated[list[BaseMessage], operator.add]:
  Without annotation → each node REPLACES the messages list
  With operator.add  → each node APPENDS to the messages list
  This ensures system prompt, memory, user question, AI responses,
  and tool results all accumulate correctly across every loop iteration.
"""
 
from __future__ import annotations
 
import operator
from typing import Annotated
 
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
 
 
class GraphState(TypedDict):
    """
    Shared state passed between all nodes in the RAG graph.
 
    messages
        Full conversation thread. Accumulates via operator.add —
        every node appends, never replaces.
        Contains: SystemMessage, HumanMessage (memory + current question),
                  AIMessage (agent responses), ToolMessage (retrieval results).
 
    question
        Raw user question stored separately so classify_node and
        collect_chunks_node can access it without scanning messages.
 
    doc_type
        'sds' or 'tds' — set by classify_node. Used by the retrieve_chunks
        tool to filter vector search to the correct document type.
 
    chunks_retrieved
        Structured chunk list extracted from ToolMessages by
        collect_chunks_node. Passed to UI for source citation display.
        Each dict: { content, source, section, score }
 
    tool_was_called
        True if the agent called retrieve_chunks at least once.
        UI uses this to warn when an answer is ungrounded.
 
    iterations
        Count of agent_node executions. Incremented each loop.
        should_continue() uses this to enforce the 5-iteration safety cap.
 
    error
        Set if graph execution fails. Surfaced in UI as an error message.
    """
 
    # ── Conversation ──────────────────────────────────────────
    messages: Annotated[list[BaseMessage], operator.add]
 
    # ── Routing ───────────────────────────────────────────────
    question: str
    doc_type: str
 
    # ── Retrieval metadata (for UI display) ───────────────────
    chunks_retrieved: list[dict]
    tool_was_called:  bool
    iterations:       int
 
    # ── Error propagation ─────────────────────────────────────
    error: str | None
 
