"""
core/tool_handler.py
─────────────────────
Defines and executes the retrieve_chunks tool used in agentic RAG.
 
Responsibilities:
  1. Provide the OpenAI tool schema (passed to chat_completion as tools=[])
  2. Execute the tool when the LLM decides to call it:
       - Embed the query
       - Run vector search in Azure AI Search with metadata filtering
       - Return structured chunk results back to the LLM
 
This is the ONLY retrieval path in the system.
The LLM decides when retrieval is needed — we never bypass this.
 
Tool contract:
  Input  → query (str), doc_type (str)
  Output → list of chunk dicts with content, source, section, score
"""
 
from __future__ import annotations
 
import json
import logging
from typing import Any
 
from services.openai_service import generate_embedding
from services.search_service import vector_search
from config.settings import TOP_K_CHUNKS
 
logger = logging.getLogger(__name__)
 
 
# ── Tool Schema ───────────────────────────────────────────────
 
RETRIEVE_CHUNKS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_chunks",
        "description": (
            "Retrieve relevant document chunks from the chemistry knowledge base. "
            "Use this tool whenever the user asks a question that requires information "
            "from uploaded SDS (Safety Data Sheet) or TDS (Technical Data Sheet) documents. "
            "Always call this before generating an answer to ground the response in source material."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A precise, self-contained search query derived from the user's question. "
                        "Rephrase follow-up questions as standalone queries using prior context."
                    ),
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["sds", "tds"],
                    "description": (
                        "The document type to search. "
                        "'sds' for safety information (hazards, PPE, handling, storage, emergency). "
                        "'tds' for technical information (specifications, performance, application)."
                    ),
                },
            },
            "required": ["query", "doc_type"],
        },
    },
}
 
 
# ── Tool Executor ─────────────────────────────────────────────
 
def execute_retrieve_chunks(
    query: str,
    doc_type: str,
    top_k: int = TOP_K_CHUNKS,
) -> list[dict[str, Any]]:
    """
    Execute the retrieve_chunks tool.
 
    Steps:
      1. Embed the query using Azure OpenAI
      2. Run vector + metadata filtered search in Azure AI Search
      3. Return structured chunk list
 
    Args:
        query:    The search query string (LLM-generated from user question).
        doc_type: "sds" or "tds" — controls metadata filter.
        top_k:    Number of chunks to retrieve (default from settings).
 
    Returns:
        List of chunk dicts, each containing:
          id, content, section, type, source, score
    """
    logger.info(
        f"[tool] retrieve_chunks called | "
        f"query='{query[:60]}...' | doc_type={doc_type}"
    )
 
    # Step 1 — Embed the query
    query_vector = generate_embedding(query)
 
    # Step 2 — Vector + metadata filtered search
    chunks = vector_search(
        query_vector=query_vector,
        doc_type=doc_type,
        top_k=top_k,
    )
 
    logger.info(f"[tool] retrieve_chunks returned {len(chunks)} chunks.")
    return chunks
 
 
# ── Tool Dispatcher ───────────────────────────────────────────
 
def dispatch_tool_call(
    tool_name: str,
    tool_arguments_json: str,
) -> str:
    """
    Dispatch an LLM tool call by name and return the result as a JSON string.
    This is called inside the agentic loop when the LLM emits a tool_call.
 
    Args:
        tool_name:            Name of the tool the LLM called (e.g. "retrieve_chunks").
        tool_arguments_json:  Raw JSON string of arguments from the LLM tool call.
 
    Returns:
        JSON string of tool results, to be sent back to the LLM as a
        tool role message.
 
    Raises:
        ValueError: If an unknown tool name is requested.
    """
    if tool_name != "retrieve_chunks":
        raise ValueError(f"Unknown tool requested by LLM: '{tool_name}'")
 
    # Parse LLM-provided arguments
    try:
        args = json.loads(tool_arguments_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool arguments: {e}")
        return json.dumps({"error": "Invalid tool arguments from LLM."})
 
    query    = args.get("query", "")
    doc_type = args.get("doc_type", "sds")
 
    if not query:
        return json.dumps({"chunks": [], "message": "Empty query — no retrieval performed."})
 
    chunks = execute_retrieve_chunks(
        query=query,
        doc_type=doc_type,
    )
 
    # Return only what the LLM needs — exclude the raw vector
    result = {
        "chunks": [
            {
                "content":          c["content"],
                "source":           c["source"],
                "section":          c.get("section", ""),
                "score":            round(c["score"], 4),
            }
            for c in chunks
        ],
        "total_retrieved": len(chunks),
    }
 
    return json.dumps(result)
 
