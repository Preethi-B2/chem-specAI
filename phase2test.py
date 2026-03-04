"""
test_phase2.py
───────────────
Integration smoke-tests for all Phase 2 Azure service wrappers.
Uses unittest.mock to patch Azure SDK calls so tests run without
real Azure credentials.
 
Run from project root:
    python test_phase2.py
 
What is tested:
  - openai_service  : generate_embedding, classify_document, classify_query
  - blob_service    : upload_document, download_document, list_user_documents
  - search_service  : ensure_index_exists, index_chunks, vector_search, get_dashboard_stats
"""
 
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
 
sys.path.insert(0, os.path.dirname(__file__))
 
 
# ══════════════════════════════════════════════════════════════
#  OpenAI Service Tests
# ══════════════════════════════════════════════════════════════
 
class TestOpenAIService(unittest.TestCase):
 
    @patch("services.openai_service._get_client")
    def test_generate_embedding_returns_vector(self, mock_get_client):
        """generate_embedding should return a list of floats."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
 
        fake_vector = [0.1, 0.2, 0.3] * 512  # 1536-dim vector
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vector)]
        )
 
        from services.openai_service import generate_embedding
        result = generate_embedding("hydrochloric acid hazards")
 
        assert isinstance(result, list), "Should return a list"
        assert len(result) == 1536, f"Expected 1536 dims, got {len(result)}"
        assert all(isinstance(v, float) for v in result), "All values should be floats"
        print("    ✅ generate_embedding: returns correct vector shape")
 
    @patch("services.openai_service._get_client")
    def test_classify_document_sds(self, mock_get_client):
        """classify_document should return 'sds' when LLM responds 'sds'."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="sds"))]
        )
 
        from services.openai_service import classify_document
        result = classify_document(
            document_text="GHS hazard classification. PPE required. OSHA section 8.",
            classifier_prompt="Classify this: {document_text}",
        )
        assert result == "sds", f"Expected 'sds', got '{result}'"
        print("    ✅ classify_document: correctly returns 'sds'")
 
    @patch("services.openai_service._get_client")
    def test_classify_document_tds(self, mock_get_client):
        """classify_document should return 'tds' when LLM responds 'tds'."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="tds"))]
        )
 
        from services.openai_service import classify_document
        result = classify_document(
            document_text="Product viscosity: 450 cP. Application temperature: 80°C.",
            classifier_prompt="Classify this: {document_text}",
        )
        assert result == "tds", f"Expected 'tds', got '{result}'"
        print("    ✅ classify_document: correctly returns 'tds'")
 
    @patch("services.openai_service._get_client")
    def test_classify_document_fallback(self, mock_get_client):
        """classify_document should default to 'tds' on unexpected LLM response."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="unknown_type"))]
        )
 
        from services.openai_service import classify_document
        result = classify_document("some text", "Classify: {document_text}")
        assert result == "tds", "Should fallback to 'tds' on unexpected response"
        print("    ✅ classify_document: fallback to 'tds' on unexpected response")
 
    @patch("services.openai_service._get_client")
    def test_classify_query(self, mock_get_client):
        """classify_query should route safety question to 'sds'."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="sds"))]
        )
 
        from services.openai_service import classify_query
        result = classify_query(
            user_question="What PPE is required for handling this chemical?",
            conversation_history="",
            query_classifier_prompt="Route: {user_question} History: {conversation_history}",
        )
        assert result == "sds"
        print("    ✅ classify_query: routes safety question to 'sds'")
 
 
# ══════════════════════════════════════════════════════════════
#  Blob Service Tests
# ══════════════════════════════════════════════════════════════
 
class TestBlobService(unittest.TestCase):
 
    @patch("services.blob_service._get_container_client")
    def test_upload_document_returns_url(self, mock_container):
        """upload_document should return the blob URL on success."""
        mock_blob_client = MagicMock()
        mock_blob_client.url = "https://mystorageaccount.blob.core.windows.net/chemistry-docs/user_abc/test.pdf"
        mock_container.return_value.get_blob_client.return_value = mock_blob_client
 
        from services.blob_service import upload_document
        url = upload_document(
            file_bytes=b"%PDF-1.4 fake content",
            blob_path="user_abc/20260304T103000_test.pdf",
        )
 
        assert url.startswith("https://"), f"URL should start with https://, got: {url}"
        assert "test.pdf" in url
        mock_blob_client.upload_blob.assert_called_once()
        print(f"    ✅ upload_document: returns blob URL: {url[:60]}...")
 
    @patch("services.blob_service._get_container_client")
    def test_download_document_returns_bytes(self, mock_container):
        """download_document should return raw bytes on success."""
        fake_bytes = b"%PDF-1.4 fake pdf content"
        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.return_value.readall.return_value = fake_bytes
        mock_container.return_value.get_blob_client.return_value = mock_blob_client
 
        from services.blob_service import download_document
        result = download_document("user_abc/test.pdf")
 
        assert result == fake_bytes
        print(f"    ✅ download_document: returns {len(result)} bytes")
 
    @patch("services.blob_service._get_container_client")
    def test_list_user_documents(self, mock_container):
        """list_user_documents should return list of blob metadata dicts."""
        from datetime import datetime, timezone
 
        mock_blob_1 = MagicMock()
        mock_blob_1.name = "user_abc/file1.pdf"
        mock_blob_1.size = 10240
        mock_blob_1.last_modified = datetime(2026, 3, 4, 10, 0, 0, tzinfo=timezone.utc)
 
        mock_blob_2 = MagicMock()
        mock_blob_2.name = "user_abc/file2.pdf"
        mock_blob_2.size = 20480
        mock_blob_2.last_modified = datetime(2026, 3, 4, 11, 0, 0, tzinfo=timezone.utc)
 
        mock_container_client = MagicMock()
        mock_container_client.list_blobs.return_value = [mock_blob_1, mock_blob_2]
        mock_container_client.url = "https://mystorageaccount.blob.core.windows.net/chemistry-docs"
        mock_container.return_value = mock_container_client
 
        from services.blob_service import list_user_documents
        results = list_user_documents("user_abc")
 
        assert len(results) == 2
        assert results[0]["name"] == "user_abc/file1.pdf"
        assert "last_modified" in results[0]
        assert "url" in results[0]
        print(f"    ✅ list_user_documents: returned {len(results)} documents")
 
# ══════════════════════════════════════════════════════════════
#  Search Service Tests
# ══════════════════════════════════════════════════════════════
 
class TestSearchService(unittest.TestCase):
 
    @patch("services.search_service._get_index_client")
    def test_ensure_index_exists_skips_if_present(self, mock_index_client):
        """ensure_index_exists should skip creation if index already exists."""
        mock_index_client.return_value.get_index.return_value = MagicMock()
 
        from services.search_service import ensure_index_exists
        ensure_index_exists()  # Should not raise
 
        mock_index_client.return_value.create_index.assert_not_called()
        print("    ✅ ensure_index_exists: skips creation if index already exists")
 
    @patch("services.search_service._get_search_client")
    def test_index_chunks_success(self, mock_search_client):
        """index_chunks should return count of successfully indexed chunks."""
        mock_result_1 = MagicMock(succeeded=True)
        mock_result_2 = MagicMock(succeeded=True)
        mock_result_3 = MagicMock(succeeded=True)
        mock_search_client.return_value.upload_documents.return_value = [
            mock_result_1, mock_result_2, mock_result_3
        ]
 
        from services.search_service import index_chunks
        chunks = [
            {
                "id":               f"chunk_{i}",
                "content":          f"Chunk {i} content about HCl hazards.",
                "section":          "Section 7 - Handling and Storage",
                "type":             "sds",                        # ← 'type' not 'doc_type'
                "source":           "hcl_sds.pdf",               # ← 'source' not 'file_name'
                "contentVector":    [0.1] * 1536,                # ← 'contentVector' not 'embedding'
                "user_id":          "user_abc",
                "upload_timestamp": "2026-03-04T10:00:00+00:00",
            }
            for i in range(3)
        ]
 
        count = index_chunks(chunks)
        assert count == 3, f"Expected 3 succeeded, got {count}"
        print(f"    ✅ index_chunks: indexed {count} chunks successfully")
 
    @patch("services.search_service._get_search_client")
    def test_index_chunks_empty(self, mock_search_client):
        """index_chunks with empty list should return 0 and not call Azure."""
        from services.search_service import index_chunks
        count = index_chunks([])
        assert count == 0
        mock_search_client.return_value.upload_documents.assert_not_called()
        print("    ✅ index_chunks: handles empty list gracefully")
 
    @patch("services.search_service._get_search_client")
    def test_vector_search_returns_chunks(self, mock_search_client):
        """vector_search should return structured chunk dicts."""
        mock_doc = {
            "id":               "chunk_001",
            "content":          "Store HCl in ventilated area away from bases.",
            "section":          "Section 7 - Handling and Storage",
            "type":             "sds",                           # ← 'type' not 'doc_type'
            "source":           "hcl_sds.pdf",                  # ← 'source' not 'file_name'
            "upload_timestamp": "2026-03-04T10:00:00+00:00",
            "@search.score":    0.92,
        }
        mock_search_client.return_value.search.return_value = [mock_doc]
 
        from services.search_service import vector_search
        results = vector_search(
            query_vector=[0.1] * 1536,
            doc_type="sds",
            user_id="user_abc",
            top_k=5,
        )
 
        assert len(results) == 1
        assert results[0]["content"] == "Store HCl in ventilated area away from bases."
        assert results[0]["score"] == 0.92
        assert results[0]["type"] == "sds"                      # ← 'type' not 'doc_type'
        assert results[0]["source"] == "hcl_sds.pdf"            # ← 'source' not 'file_name'
        assert results[0]["section"] == "Section 7 - Handling and Storage"
        print(f"    ✅ vector_search: returned {len(results)} chunk(s) with correct structure")
 
    @patch("services.search_service._get_search_client")
    def test_get_dashboard_stats(self, mock_search_client):
        """get_dashboard_stats should return a dict with all required keys."""
        mock_total = MagicMock()
        mock_total.get_count.return_value = 42
        mock_sds = MagicMock()
        mock_sds.get_count.return_value = 28
        mock_tds = MagicMock()
        mock_tds.get_count.return_value = 14
 
        mock_recent_doc = {
            "source":           "hcl_sds.pdf",                  # ← 'source' not 'file_name'
            "type":             "sds",                           # ← 'type' not 'doc_type'
            "upload_timestamp": "2026-03-04T10:00:00+00:00",
        }
        mock_recent = MagicMock()
        mock_recent.get_count.return_value = 0
        mock_recent.__iter__ = MagicMock(return_value=iter([mock_recent_doc]))
 
        mock_search_client.return_value.search.side_effect = [
            mock_total, mock_sds, mock_tds, mock_recent
        ]
 
        from services.search_service import get_dashboard_stats
        stats = get_dashboard_stats("user_abc")
 
        required_keys = {"total_chunks", "sds_count", "tds_count", "unique_files", "recent_uploads"}
        assert required_keys.issubset(stats.keys()), f"Missing keys: {required_keys - stats.keys()}"
        assert stats["total_chunks"] == 42
        assert stats["sds_count"] == 28
        assert stats["tds_count"] == 14
        # Verify recent_uploads uses 'source' and 'type' keys (not 'file_name'/'doc_type')
        if stats["recent_uploads"]:
            upload = stats["recent_uploads"][0]
            assert "source" in upload, "recent_uploads should use 'source' key"
            assert "type" in upload, "recent_uploads should use 'type' key"
        print(f"    ✅ get_dashboard_stats: {stats}")
 
 
# ══════════════════════════════════════════════════════════════
#  Main Runner
# ══════════════════════════════════════════════════════════════
 
def main():
    print("=" * 55)
    print("  Phase 2 Smoke Tests — Azure Service Wrappers")
    print("=" * 55)
 
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
 
    print("\n[1] OpenAI Service Tests")
    suite.addTests(loader.loadTestsFromTestCase(TestOpenAIService))
 
    print("\n[2] Blob Service Tests")
    suite.addTests(loader.loadTestsFromTestCase(TestBlobService))
 
    print("\n[3] Search Service Tests")
    suite.addTests(loader.loadTestsFromTestCase(TestSearchService))
 
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
    result = runner.run(suite)
 
    # Re-run with our own print statements
    for test_class in [TestOpenAIService, TestBlobService, TestSearchService]:
        for method_name in [m for m in dir(test_class) if m.startswith("test_")]:
            test = test_class(method_name)
            try:
                test.debug()
            except Exception as e:
                print(f"    ❌ {method_name}: {e}")
 
    if result.wasSuccessful():
        print("\n" + "=" * 55)
        print("  ✅ All Phase 2 tests passed!")
        print("=" * 55)
    else:
        print(f"\n❌ {len(result.failures)} failure(s), {len(result.errors)} error(s)")
        for _, tb in result.failures + result.errors:
            print(tb)
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()
 