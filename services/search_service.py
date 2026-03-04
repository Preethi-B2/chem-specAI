"""
services/search_service.py
───────────────────────────
Azure AI Search wrapper for:
  - Index creation (with vector field schema)
  - Indexing document chunks with embeddings + metadata
  - Vector + metadata filtered search
  - Dashboard analytics queries (counts, recent uploads)
 
Uses azure-search-documents>=11.6.0 with the SearchClient and
SearchIndexClient APIs.
 
── Index Field Schema (exact Azure index field names) ────────
  id              → Unique chunk identifier (key field)
  content         → Chunk text
  section         → Section heading / label within the document
  type            → Document type: "sds" or "tds"
  source          → Source filename, e.g. "chemical_x_sds.pdf"
  contentVector   → Embedding vector (HNSW)
"""
 
from __future__ import annotations
 
import logging
from typing import Any
 
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchField,
)
from azure.search.documents.models import VectorizedQuery
 
from config.settings import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_OPENAI_EMBEDDING_DIMENSIONS,
    TOP_K_CHUNKS,
)
 
logger = logging.getLogger(__name__)
 
# ── Credential ────────────────────────────────────────────────
_CREDENTIAL = AzureKeyCredential(AZURE_SEARCH_API_KEY)
 
 
# ── Clients ───────────────────────────────────────────────────
 
def _get_index_client() -> SearchIndexClient:
    """Return a SearchIndexClient for index management operations."""
    return SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=_CREDENTIAL,
    )
 
 
def _get_search_client() -> SearchClient:
    """Return a SearchClient for document indexing and querying."""
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=_CREDENTIAL,
    )
 
 
# ── Index Bootstrap ───────────────────────────────────────────
 
def ensure_index_exists() -> None:
    """
    Create the Azure AI Search index if it does not already exist.
 
    Index schema (exact field names used in Azure):
      id, content, section, type, source, contentVector,
 
    Safe to call on every app startup — skips creation if index exists.
    """
    index_client = _get_index_client()
 
    try:
        index_client.get_index(AZURE_SEARCH_INDEX_NAME)
        logger.debug(f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists.")
        return
    except ResourceNotFoundError:
        logger.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' not found. Creating...")
 
    # HNSW vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-config"),
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )
 
    fields = [
        # ── Key field ────────────────────────────────────────
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        # ── Text content ─────────────────────────────────────
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        # ── Section label within document ────────────────────
        SearchableField(
            name="section",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        # ── Document type: "sds" or "tds" ────────────────────
        SimpleField(
            name="type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        # ── Source filename ───────────────────────────────────
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        # ── Embedding vector ──────────────────────────────────
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=AZURE_OPENAI_EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
        # ── User isolation ────────────────────────────────────
        SimpleField(
            name="user_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
    ]
 
    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )
 
    index_client.create_index(index)
    logger.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' created successfully.")
 
 
# ── Index Chunks ─────────────────────────────────────────────
 
def index_chunks(chunks: list[dict[str, Any]]) -> int:
    """
    Upload a batch of document chunks (with embeddings) to Azure AI Search.
 
    Each chunk dict must match the index schema exactly:
        {
            "id":               str,   # Unique chunk ID
            "content":          str,   # Chunk text
            "section":          str,   # Section label (e.g. "Section 7 - Storage")
            "type":             str,   # "sds" or "tds"
            "source":           str,   # Source filename
            "contentVector":    list,  # Embedding vector
        }
 
    Args:
        chunks: List of chunk dicts to index.
 
    Returns:
        Number of chunks successfully indexed.
 
    Raises:
        AzureError: On indexing failure.
    """
    if not chunks:
        logger.warning("index_chunks called with empty list — nothing to index.")
        return 0
 
    try:
        search_client = _get_search_client()
        results = search_client.upload_documents(documents=chunks)
 
        succeeded = sum(1 for r in results if r.succeeded)
        failed = len(results) - succeeded
 
        if failed > 0:
            logger.warning(f"Indexed {succeeded}/{len(chunks)} chunks. {failed} failed.")
        else:
            logger.info(f"Successfully indexed {succeeded} chunks.")
 
        return succeeded
 
    except AzureError as e:
        logger.error(f"Failed to index chunks: {e}")
        raise
 
 
# ── Vector Search ─────────────────────────────────────────────
 
def vector_search(
    query_vector: list[float],
    doc_type: str,
    top_k: int = TOP_K_CHUNKS,
) -> list[dict[str, Any]]:
    """
    Perform vector similarity search with metadata filtering.
 
    Combines:
      - HNSW vector search on the 'contentVector' field
      - OData filter on 'type' (sds/tds)
 
    Args:
        query_vector: Embedding of the user's query.
        doc_type:     "sds" or "tds" — maps to the 'type' field in index.
        top_k:        Number of top chunks to retrieve.
 
    Returns:
        List of chunk dicts with keys:
          id, content, section, type, source, score
    """
    try:
        search_client = _get_search_client()
 
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="contentVector",              # ← exact index field name
        )
 
        # OData filter using exact index field names
        odata_filter = f"type eq '{doc_type}'"
 
        results = search_client.search(
            search_text=None,                    # Pure vector search
            vector_queries=[vector_query],
            filter=odata_filter,
            select=["id", "content", "section", "type", "source"],
            top=top_k,
        )
 
        chunks = []
        for result in results:
            chunks.append({
                "id":               result["id"],
                "content":          result["content"],
                "section":          result.get("section", ""),
                "type":             result["type"],
                "source":           result["source"],
                "score":            result["@search.score"],
            })
 
        logger.info(
            f"Vector search returned {len(chunks)} chunks "
            f"[type={doc_type}]"
        )
        return chunks
 
    except AzureError as e:
        logger.error(f"Vector search failed: {e}")
        raise
 
 
# ── Dashboard Analytics ───────────────────────────────────────
 
def get_dashboard_stats() -> dict[str, Any]:
    """
    Fetch analytics data for the dashboard tab.
    Uses exact index field names: 'type', 'source'.
 
    Returns:
        Dict with keys:
          total_chunks, sds_count, tds_count,
          unique_files, recent_uploads (list of dicts)
    """
    try:
        search_client = _get_search_client()
 
        # ── Total chunks ──────────────────────────────────────
        total_result = search_client.search(
            search_text="*",
            include_total_count=True,
            top=0,
        )
        total_chunks = total_result.get_count() or 0
 
        # ── SDS chunk count ───────────────────────────────────
        sds_result = search_client.search(
            search_text="*",
            filter="type eq 'sds'",
            include_total_count=True,
            top=0,
        )
        sds_count = sds_result.get_count() or 0
 
        # ── TDS chunk count ───────────────────────────────────
        tds_result = search_client.search(
            search_text="*",
            filter="type eq 'tds'",
            include_total_count=True,
            top=0,
        )
        tds_count = tds_result.get_count() or 0
 
        # ── Recent uploads (deduplicated by source) ───────────
        recent_result = search_client.search(
            search_text="*",
            select=["source", "type"],
            top=50,
        )
 
        seen_files: set[str] = set()
        recent_uploads: list[dict] = []
        for doc in recent_result:
            fname = doc.get("source", "")
            if fname not in seen_files:
                seen_files.add(fname)
                recent_uploads.append({
                    "source":           fname,
                    "type":             doc.get("type", ""),
                })
            if len(recent_uploads) >= 10:
                break
 
        return {
            "total_chunks":   total_chunks,
            "sds_count":      sds_count,
            "tds_count":      tds_count,
            "unique_files":   len(seen_files),
            "recent_uploads": recent_uploads,
        }
 
    except AzureError as e:
        logger.error(f"Dashboard stats query failed: {e}")
        raise
 
