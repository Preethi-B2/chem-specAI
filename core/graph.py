"""
core/graph.py
──────────────
Builds the LangGraph RAG pipeline and exposes run_query().
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass
 
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
    _OUT_OF_SCOPE_MESSAGE,
    _ETHICAL_REFUSAL_MESSAGE,
)
 
from utils.prompt_loader import load_prompt
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class QueryResult:
    answer: str
    doc_type: str
    chunks_retrieved: list[dict]
    tool_was_called: bool
    iterations: int
    error: str | None = None
 
 
def _build_graph():
 
    g = StateGraph(GraphState)
 
    g.add_node("classify", classify_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("collect_chunks", collect_chunks_node)
 
    g.set_entry_point("classify")
 
    g.add_edge("classify", "agent")
    g.add_edge("tools", "agent")
    g.add_edge("collect_chunks", END)
 
    g.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": "collect_chunks",
        },
    )
 
    return g.compile()
 
 
_graph = _build_graph()
 
 
def run_query(
    user_question: str,
    user_id: str,
    memory_messages: list[dict],
    conversation_history_text: str,
) -> QueryResult:
 
    logger.info(f"[run_query] Starting graph for: '{user_question[:60]}...'")
 
    lc_memory: list = []
 
    for m in memory_messages:
        if m["role"] == "user":
            lc_memory.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_memory.append(AIMessage(content=m["content"]))
 
    system_prompt = load_prompt("system_prompt.md")
 
    initial_state: GraphState = {
        "messages": [
            SystemMessage(content=system_prompt),
            *lc_memory,
            HumanMessage(content=user_question),
        ],
        "question": user_question,
        "doc_type": "sds",
        "chunks_retrieved": [],
        "tool_was_called": False,
        "iterations": 0,
        "error": None,
    }
 
    try:
        final_state: GraphState = _graph.invoke(initial_state)
 
    except Exception as e:
        logger.exception("[run_query] Graph execution failed.")
 
        return QueryResult(
            answer=_OUT_OF_SCOPE_MESSAGE,
            doc_type="sds",
            chunks_retrieved=[],
            tool_was_called=False,
            iterations=0,
            error=str(e),
        )
 
    ai_messages = [
        m for m in final_state["messages"]
        if isinstance(m, AIMessage) and m.content
    ]
 
    if ai_messages:
        final_answer = ai_messages[-1].content
    else:
        final_answer = (
            "I was unable to generate an answer. Please try rephrasing your question."
        )
 
    # ── Replace prompt placeholders if they appear ─────────────
    if "{{_SCOPE_REFUSAL}}" in final_answer:
        final_answer = _OUT_OF_SCOPE_MESSAGE
 
    if "{{_ETHICAL_REFUSAL}}" in final_answer:
        final_answer = _ETHICAL_REFUSAL_MESSAGE
 
    logger.info(
        f"[run_query] Completed in {final_state.get('iterations', 0)} iteration(s). "
        f"Chunks: {len(final_state.get('chunks_retrieved', []))}. "
        f"doc_type: {final_state.get('doc_type', 'unknown')}."
    )
 
    return QueryResult(
        answer=final_answer,
        doc_type=final_state.get("doc_type", "sds"),
        chunks_retrieved=final_state.get("chunks_retrieved", []),
        tool_was_called=final_state.get("tool_was_called", False),
        iterations=final_state.get("iterations", 0),
        error=final_state.get("error"),
    )