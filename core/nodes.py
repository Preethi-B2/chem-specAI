"""
core/nodes.py
──────────────
All node functions for the LangGraph RAG pipeline.
"""
 
from __future__ import annotations
 
import json
import logging
 
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
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
 
 
# ── Refusal messages ──────────────────────────────────────────
 
# Harmful / unethical requests
_ETHICAL_REFUSAL_MESSAGE = (
    "I’m designed to operate in an ethical and responsible way, "
    "so I can’t assist with that request."
)
 
# Questions outside SDS/TDS scope
_OUT_OF_SCOPE_MESSAGE = (
    "I'm designed to help with SDS and TDS chemistry documents. "
    "Please ask a question related to those documents."
)
 
 
# ── Shared LLM factory ────────────────────────────────────────
 
def _get_llm() -> AzureChatOpenAI:
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
    logger.info("[classify_node] classifying question...")
 
    try:
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
 
    except Exception as e:
        logger.error(f"[classify_node] Classification failed: {e}")
        doc_type = "sds"
 
    logger.info(f"[classify_node] doc_type = '{doc_type}'")
 
    return {"doc_type": doc_type}
 
 
# ── Node 2: agent_node ────────────────────────────────────────
 
def agent_node(state: GraphState) -> dict:
    iteration = state.get("iterations", 0) + 1
    logger.info(f"[agent_node] iteration {iteration}/{_MAX_ITERATIONS}")
 
    try:
        llm = _get_llm()
        response: AIMessage = llm.invoke(state["messages"])
 
    except Exception as e:
        error_str = str(e).lower()
 
        if any(keyword in error_str for keyword in [
            "content_filter", "content filter", "filtered",
            "responsible ai", "safety", "policy", "violat",
        ]):
            logger.warning(f"[agent_node] Ethical content blocked: {e}")
            refusal = _ETHICAL_REFUSAL_MESSAGE
        else:
            logger.error(f"[agent_node] LLM call failed: {e}")
            refusal = _OUT_OF_SCOPE_MESSAGE
 
        return {
            "messages": [AIMessage(content=refusal)],
            "iterations": iteration,
        }
 
    return {
        "messages": [response],
        "iterations": iteration,
    }
 
 
# ── Node 3: collect_chunks_node ───────────────────────────────
 
def collect_chunks_node(state: GraphState) -> dict:
    logger.info("[collect_chunks_node] extracting chunks from ToolMessages...")
 
    all_chunks: list[dict] = []
 
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(msg.content)
                all_chunks.extend(result.get("chunks", []))
            except Exception:
                logger.warning(
                    "[collect_chunks_node] Failed to parse ToolMessage content."
                )
 
    logger.info(
        f"[collect_chunks_node] total chunks collected: {len(all_chunks)}"
    )
 
    return {
        "chunks_retrieved": all_chunks,
        "tool_was_called": len(all_chunks) > 0,
    }
 
 
# ── Conditional edge ──────────────────────────────────────────
 
def should_continue(state: GraphState) -> str:
 
    if state.get("iterations", 0) >= _MAX_ITERATIONS:
        logger.warning("[should_continue] Max iterations reached.")
        return "end"
 
    last_message = state["messages"][-1]
 
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("[should_continue] → tools")
        return "tools"
 
    logger.info("[should_continue] → end")
    return "end"