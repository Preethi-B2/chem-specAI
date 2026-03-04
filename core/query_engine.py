"""
core/query_engine.py
─────────────────────
Orchestrates the full Chat query pipeline (Tab 2 flow):
 
  User question
    → Load short-term memory
    → Classify query as SDS or TDS
    → Build messages[] with system prompt + memory + question
    → Call LLM with retrieve_chunks tool attached
    → If LLM calls the tool → execute it → feed results back
    → LLM generates final grounded answer
    → Caller updates memory with the turn
 
This is the agentic RAG loop. The LLM controls when retrieval happens.
We never bypass tool calling to call vector_search directly.
 
IMPORTANT RULE:
  Memory provides context — it never replaces retrieval.
  Every answer must be grounded in retrieved chunks.
"""
 
from __future__ import annotations
 
import json
import logging
from dataclasses import dataclass, field
from typing import Any
 
from services.openai_service import (
    chat_completion,
    classify_query,
)
from core.tool_handler import (
    RETRIEVE_CHUNKS_TOOL,
    dispatch_tool_call,
)
from utils.prompt_loader import load_prompt
 
logger = logging.getLogger(__name__)
 
# Maximum tool call iterations to prevent infinite loops
_MAX_TOOL_ITERATIONS = 5
 
 
# ── Result Model ──────────────────────────────────────────────
 
@dataclass
class QueryResult:
    """
    Full result of a chat query, returned to the UI.
    """
    answer: str                          # Final LLM-generated answer
    doc_type: str                        # "sds" or "tds" (classified)
    chunks_retrieved: list[dict]         # Chunks returned by retrieve_chunks
    tool_was_called: bool                # Whether the LLM invoked retrieval
    iterations: int                      # How many LLM roundtrips were needed
    error: str | None = None             # Set if something went wrong
 
 
# ── Query Classification ──────────────────────────────────────
 
def classify_user_query(
    user_question: str,
    conversation_history_text: str,
) -> str:
    """
    Classify the user's question as "sds" or "tds" using the LLM.
    Used to set the doc_type filter for retrieval.
 
    Args:
        user_question:            The current user question.
        conversation_history_text: Formatted prior turns (from memory_manager).
 
    Returns:
        "sds" or "tds"
    """
    query_classifier_prompt = load_prompt("query_classifier.md")
    return classify_query(
        user_question=user_question,
        conversation_history=conversation_history_text,
        query_classifier_prompt=query_classifier_prompt,
    )
 
 
# ── Agentic Tool-Calling Loop ─────────────────────────────────
 
def run_query(
    user_question: str,
    user_id: str,
    memory_messages: list[dict[str, str]],
    conversation_history_text: str,
) -> QueryResult:
    """
    Run the full agentic query pipeline for a single user question.
 
    Flow:
      1. Classify query → doc_type
      2. Build messages[] = system_prompt + memory + user question
      3. Call LLM with retrieve_chunks tool
      4. If LLM calls tool → execute → append tool result → call LLM again
      5. Repeat until LLM produces a text answer (no more tool calls)
      6. Return QueryResult with answer + metadata
 
    Args:
        user_question:             The current user's question string.
        user_id:                   Current session user ID for retrieval isolation.
        memory_messages:           List of prior {"role", "content"} dicts from memory_manager.
        conversation_history_text: Plain-text version of memory for prompt placeholders.
 
    Returns:
        QueryResult with the final answer and retrieval metadata.
    """
 
    # ── Step 1: Classify query ────────────────────────────────
    doc_type = classify_user_query(user_question, conversation_history_text)
    logger.info(f"Query classified as: '{doc_type}'")
 
    # ── Step 2: Build initial messages[] ─────────────────────
    system_prompt = load_prompt("system_prompt.md")
 
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *memory_messages,                              # Inject short-term memory
        {"role": "user", "content": user_question},
    ]
 
    # ── Step 3–5: Agentic loop ────────────────────────────────
    all_chunks: list[dict] = []
    tool_was_called = False
    iterations = 0
 
    while iterations < _MAX_TOOL_ITERATIONS:
        iterations += 1
        logger.info(f"LLM call iteration {iterations}/{_MAX_TOOL_ITERATIONS}")
 
        response = chat_completion(
            messages=messages,
            tools=[RETRIEVE_CHUNKS_TOOL],
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1500,
        )
 
        response_message = response.choices[0].message
 
        # ── Check if LLM wants to call a tool ────────────────
        if response_message.tool_calls:
            tool_was_called = True
            messages.append(response_message)   # Append assistant's tool_call message
 
            # Execute each tool call and append results
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args_json = tool_call.function.arguments
 
                logger.info(f"LLM called tool: '{tool_name}'")
 
                # Execute — server-side user_id is injected for security
                tool_result_json = dispatch_tool_call(
                    tool_name=tool_name,
                    tool_arguments_json=tool_args_json,
                    user_id=user_id,
                )
 
                # Collect retrieved chunks for the QueryResult metadata
                try:
                    tool_result = json.loads(tool_result_json)
                    all_chunks.extend(tool_result.get("chunks", []))
                except json.JSONDecodeError:
                    pass
 
                # Append tool result back into messages for next LLM call
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      tool_result_json,
                })
 
            # Continue loop — LLM will now see the tool results
            continue
 
        # ── LLM produced a text answer (no tool call) ─────────
        final_answer = response_message.content or ""
 
        # If tool was never called, the LLM answered from memory/general knowledge.
        # This can happen for greetings or out-of-scope questions.
        if not tool_was_called:
            logger.warning(
                "LLM answered without calling retrieve_chunks. "
                "Answer may not be grounded in documents."
            )
 
        logger.info(f"Query completed in {iterations} iteration(s).")
        return QueryResult(
            answer=final_answer,
            doc_type=doc_type,
            chunks_retrieved=all_chunks,
            tool_was_called=tool_was_called,
            iterations=iterations,
        )
 
    # ── Safety net: loop limit hit ────────────────────────────
    logger.error(f"Tool call loop limit ({_MAX_TOOL_ITERATIONS}) reached without a final answer.")
    return QueryResult(
        answer=(
            "I was unable to generate a complete answer after multiple retrieval attempts. "
            "Please try rephrasing your question."
        ),
        doc_type=doc_type,
        chunks_retrieved=all_chunks,
        tool_was_called=tool_was_called,
        iterations=iterations,
        error="Max tool call iterations reached.",
    )
 
