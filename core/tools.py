"""
core/tools.py
──────────────
Defines the retrieve_chunks tool using LangGraph's @tool decorator.
 
Replaces:
  - RETRIEVE_CHUNKS_TOOL dict    (from old tool_handler.py)
  - execute_retrieve_chunks()    (from old tool_handler.py)
  - dispatch_tool_call()         (from old tool_handler.py)
 
The @tool decorator does three things automatically:
  1. Reads the docstring        → tool description sent to the LLM
  2. Reads the type hints       → JSON parameter schema for the LLM
  3. Wraps the function         → ToolNode can invoke it automatically
 
No manual JSON schema, no dispatcher, no argument parsing needed.
"""
 
from __future__ import annotations
 
import logging
 
from langchain_core.tools import tool
 
from services.openai_service import generate_embedding
from services.search_service import vector_search
from config.settings import TOP_K_CHUNKS
 
logger = logging.getLogger(__name__)
 
 
@tool
def retrieve_chunks(query: str, doc_type: str) -> dict:
    """
    Retrieve relevant document chunks from the chemistry knowledge base.
 
    Use this tool whenever the user asks a question that requires information
    from uploaded SDS (Safety Data Sheet) or TDS (Technical Data Sheet) documents.
    Always call this before generating an answer to ground the response in
    source material.
 
    Args:
        query:    A precise, self-contained search query derived from the user's
                  question. Rephrase follow-up questions as standalone queries
                  using prior context.
        doc_type: The document type to search.
                  'sds' for safety information — hazards, PPE, handling,
                  storage, exposure limits, emergency procedures.
                  'tds' for technical information — specifications,
                  performance data, application methods, composition.
    """
    logger.info(
        f"[tool] retrieve_chunks | "
        f"doc_type={doc_type} | query='{query[:60]}...'"
    )
 
    # Step 1 — Embed the query using Azure OpenAI
    query_vector = generate_embedding(query)
 
    # Step 2 — Vector search filtered by doc_type
    chunks = vector_search(
        query_vector=query_vector,
        doc_type=doc_type,
        top_k=TOP_K_CHUNKS,
    )
 
    logger.info(f"[tool] retrieve_chunks returned {len(chunks)} chunks.")
 
    # Return only what the LLM needs — no raw vectors
    return {
        "chunks": [
            {
                "content": c["content"],
                "source":  c["source"],
                "section": c.get("section", ""),
                "score":   round(c["score"], 4),
            }
            for c in chunks
        ],
        "total_retrieved": len(chunks),
    }
 
 
# Exported list — used by nodes.py (_get_llm().bind_tools)
# and graph.py (ToolNode)
TOOLS = [retrieve_chunks]
