"""
core/nodes.py
──────────────
All node functions for the LangGraph RAG pipeline.
 
Each node is a pure function:
  Input  → GraphState (read-only)
  Output → dict of fields to merge back into GraphState
 
LangGraph calls these functions in the order defined by the graph
edges in graph.py. Each return dict is merged into the shared state
automatically — nodes never mutate state directly.
 
Nodes defined here:
  classify_node       — classify question as 'sds' or 'tds'
  agent_node          — call LLM (decides: answer or call tool)
  collect_chunks_node — extract retrieved chunks from ToolMessages
  should_continue     — conditional edge: 'tools' or 'end'
 
tool_node is built using LangGraph's prebuilt ToolNode in graph.py.
"""
 
from __future__ import annotations
 
import json
import logging
 
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
 
from core.state import GraphState
from core.tools import TOOLS
from services.openai_service import classify_query
from utils.prompt_loader import load_prompt
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
)
 
logger = logging.getLogger(__name__)
 
# Safety cap — max agent_node iterations before forcing END
_MAX_ITERATIONS = 5
 
 
# ── Shared LLM factory ────────────────────────────────────────
 
def _get_llm() -> AzureChatOpenAI:
    """
    Return an AzureChatOpenAI instance with retrieve_chunks bound.
 
    .bind_tools() attaches the tool schema so the LLM can emit
    tool_calls in its response. Called fresh per agent_node execution
    to stay stateless.
    """
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        temperature=0.0,
        max_tokens=1500,
    )
    return llm.bind_tools(TOOLS)
 
 
# ── Node 1: classify_node ─────────────────────────────────────
 
def classify_node(state: GraphState) -> dict:
    """
    Classify the user's question as 'sds' or 'tds'.
 
    Uses the query_classifier.md prompt + conversation history
    to determine which document type the question is about.
    Sets doc_type in state, which the retrieve_chunks tool reads
    to apply the correct OData filter on vector search.
 
    Returns:
        { doc_type: 'sds' | 'tds' }
    """
    logger.info("[classify_node] classifying question...")
 
    # Build conversation history text from prior HumanMessage/AIMessage pairs
    history_lines = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            history_lines.append(f"Assistant: {msg.content}")
    conversation_history = "\n".join(history_lines) if history_lines else ""
 
    prompt = load_prompt("query_classifier.md")
    doc_type = classify_query(
        user_question=state["question"],
        conversation_history=conversation_history,
        query_classifier_prompt=prompt,
    )
 
    logger.info(f"[classify_node] doc_type = '{doc_type}'")
    return {"doc_type": doc_type}
 
 
# ── Node 2: agent_node ────────────────────────────────────────
 
def agent_node(state: GraphState) -> dict:
    """
    Call the LLM with the full message history and tools attached.
 
    The LLM will either:
      a) Emit tool_calls → retrieve_chunks will be executed by tool_node
      b) Return a text answer → should_continue routes to END
 
    Increments the iteration counter on every call.
 
    Returns:
        { messages: [AIMessage], iterations: int }
    """
    iteration = state.get("iterations", 0) + 1
    logger.info(f"[agent_node] iteration {iteration}/{_MAX_ITERATIONS}")
 
    llm = _get_llm()
    response: AIMessage = llm.invoke(state["messages"])
 
    return {
        "messages":   [response],   # operator.add appends to existing messages
        "iterations": iteration,
    }
 
 
# ── Node 3: collect_chunks_node ───────────────────────────────
 
def collect_chunks_node(state: GraphState) -> dict:
    """
    Extract retrieved chunks from all ToolMessages in the message history.
 
    Runs once after the agent loop ends. Parses every ToolMessage
    in the accumulated messages list to collect all chunks returned
    by retrieve_chunks across all iterations.
 
    Returns:
        { chunks_retrieved: list[dict], tool_was_called: bool }
    """
    logger.info("[collect_chunks_node] extracting chunks from ToolMessages...")
 
    all_chunks: list[dict] = []
 
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                # ToolMessage.content is the JSON string returned by retrieve_chunks
                result = json.loads(msg.content)
                all_chunks.extend(result.get("chunks", []))
            except (json.JSONDecodeError, AttributeError, TypeError):
                logger.warning(
                    f"[collect_chunks_node] Failed to parse ToolMessage content."
                )
 
    logger.info(
        f"[collect_chunks_node] total chunks collected: {len(all_chunks)}"
    )
 
    return {
        "chunks_retrieved": all_chunks,
        "tool_was_called":  len(all_chunks) > 0,
    }
 
 
# ── Conditional edge: should_continue ────────────────────────
 
def should_continue(state: GraphState) -> str:
    """
    Decide which node executes after agent_node.
 
    Called by LangGraph as a conditional edge function.
    Must return a string key that matches the edge map in graph.py.
 
    Logic:
      - If iteration cap reached → force 'end' (safety valve)
      - If last message has tool_calls → route to 'tools'
      - Otherwise → route to 'end' (LLM gave a text answer)
 
    Returns:
        'tools' → execute tool_node next
        'end'   → execute collect_chunks_node next, then END
    """
    # Safety cap
    if state.get("iterations", 0) >= _MAX_ITERATIONS:
        logger.warning(
            f"[should_continue] Max iterations ({_MAX_ITERATIONS}) reached. "
            "Forcing end."
        )
        return "end"
 
    # Check if the last message is an AIMessage with tool_calls
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("[should_continue] → tools")
        return "tools"
 
    logger.info("[should_continue] → end")
    return "end"