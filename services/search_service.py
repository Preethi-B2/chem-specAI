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
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchField as VectorField,
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
 
    Index schema matches the chunk data model:
      id, content, embedding, doc_type, file_name, user_id, upload_timestamp
 
    Safe to call on every app startup — will skip creation if index exists.
    """
    index_client = _get_index_client()
 
    try:
        index_client.get_index(AZURE_SEARCH_INDEX_NAME)
        logger.debug(f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists.")
        return
    except ResourceNotFoundError:
        logger.info(f"Index '{AZURE_SEARCH_INDEX_NAME}' not found. Creating...")
 
    # Vector search configuration
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
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        VectorField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=AZURE_OPENAI_EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
        SimpleField(
            name="doc_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="file_name",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        SimpleField(
            name="user_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="upload_timestamp",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            retrievable=True,
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
 
    Each chunk dict must contain:
        id, content, embedding, doc_type, file_name, user_id, upload_timestamp
 
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
    user_id: str,
    top_k: int = TOP_K_CHUNKS,
) -> list[dict[str, Any]]:
    """
    Perform vector similarity search with metadata filtering.
 
    Combines:
      - HNSW vector search on the 'embedding' field
      - OData filter for doc_type and user_id isolation
 
    Args:
        query_vector: Embedding of the user's query.
        doc_type:     "sds" or "tds" — filters retrieval to correct domain.
        user_id:      Current user's ID — enforces per-user isolation.
        top_k:        Number of top chunks to retrieve.
 
    Returns:
        List of chunk dicts with keys: id, content, doc_type, file_name,
        upload_timestamp, and @search.score.
    """
    try:
        search_client = _get_search_client()
 
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )
 
        # OData filter: enforce domain + user isolation
        odata_filter = (
            f"doc_type eq '{doc_type}' and user_id eq '{user_id}'"
        )
 
        results = search_client.search(
            search_text=None,  # Pure vector search (no keyword mixing)
            vector_queries=[vector_query],
            filter=odata_filter,
            select=["id", "content", "doc_type", "file_name", "upload_timestamp"],
            top=top_k,
        )
 
        chunks = []
        for result in results:
            chunks.append({
                "id": result["id"],
                "content": result["content"],
                "doc_type": result["doc_type"],
                "file_name": result["file_name"],
                "upload_timestamp": result.get("upload_timestamp", ""),
                "score": result["@search.score"],
            })
 
        logger.info(
            f"Vector search returned {len(chunks)} chunks "
            f"[doc_type={doc_type}, user_id={user_id[:8]}...]"
        )
        return chunks
 
    except AzureError as e:
        logger.error(f"Vector search failed: {e}")
        raise
 
 
# ── Dashboard Analytics ───────────────────────────────────────
 
def get_dashboard_stats(user_id: str) -> dict[str, Any]:
    """
    Fetch analytics data for the dashboard tab.
    All queries are scoped to the current user_id.
 
    Args:
        user_id: Current user's session identifier.
 
    Returns:
        Dict with keys:
          total_chunks, sds_count, tds_count,
          unique_files, recent_uploads (list of dicts)
    """
    try:
        search_client = _get_search_client()
        user_filter = f"user_id eq '{user_id}'"
 
        # Total chunks
        total_result = search_client.search(
            search_text="*",
            filter=user_filter,
            include_total_count=True,
            top=0,
        )
        total_chunks = total_result.get_count() or 0
 
        # SDS chunks
        sds_result = search_client.search(
            search_text="*",
            filter=f"{user_filter} and doc_type eq 'sds'",
            include_total_count=True,
            top=0,
        )
        sds_count = sds_result.get_count() or 0
 
        # TDS chunks
        tds_result = search_client.search(
            search_text="*",
            filter=f"{user_filter} and doc_type eq 'tds'",
            include_total_count=True,
            top=0,
        )
        tds_count = tds_result.get_count() or 0
 
        # Recent uploads — top 10 by timestamp, distinct file names
        recent_result = search_client.search(
            search_text="*",
            filter=user_filter,
            select=["file_name", "doc_type", "upload_timestamp"],
            order_by=["upload_timestamp desc"],
            top=50,  # Fetch more, deduplicate below
        )
 
        seen_files: set[str] = set()
        recent_uploads: list[dict] = []
        for doc in recent_result:
            fname = doc.get("file_name", "")
            if fname not in seen_files:
                seen_files.add(fname)
                recent_uploads.append({
                    "file_name": fname,
                    "doc_type": doc.get("doc_type", ""),
                    "upload_timestamp": doc.get("upload_timestamp", ""),
                })
            if len(recent_uploads) >= 10:
                break
 
        return {
            "total_chunks": total_chunks,
            "sds_count": sds_count,
            "tds_count": tds_count,
            "unique_files": len(seen_files),
            "recent_uploads": recent_uploads,
        }
 
    except AzureError as e:
        logger.error(f"Dashboard stats query failed: {e}")
        raise