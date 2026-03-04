"""
test_phase3.py
───────────────
Unit tests for all Phase 3 core business logic modules.
All Azure SDK and service calls are mocked — no credentials needed.
 
Run from project root:
    python test_phase3.py
 
Modules tested:
  - core/document_processor.py
  - core/memory_manager.py
  - core/tool_handler.py
  - core/query_engine.py
"""
 
from __future__ import annotations
 
import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch
 
sys.path.insert(0, os.path.dirname(__file__))
 
 
# ══════════════════════════════════════════════════════════════
#  document_processor Tests
# ══════════════════════════════════════════════════════════════
 
class TestDocumentProcessor(unittest.TestCase):
 
    def test_extract_text_empty_pdf_raises(self):
        """extract_text_from_pdf should raise ValueError for text-less PDFs."""
        from core.document_processor import extract_text_from_pdf
        import pypdf
        from io import BytesIO
        from unittest.mock import patch, MagicMock
 
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
 
        with patch("core.document_processor.pypdf.PdfReader") as mock_reader:
            mock_reader.return_value.pages = [mock_page]
            with self.assertRaises(ValueError) as ctx:
                extract_text_from_pdf(b"fake pdf bytes")
            assert "No extractable text" in str(ctx.exception)
        print("    ✅ extract_text_from_pdf: raises ValueError for empty PDF")
 
    def test_extract_text_success(self):
        """extract_text_from_pdf should return text and page count."""
        from core.document_processor import extract_text_from_pdf
        from unittest.mock import patch, MagicMock
 
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "SECTION 1: IDENTIFICATION. Product: HCl."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "SECTION 2: HAZARD IDENTIFICATION. Danger."
 
        with patch("core.document_processor.pypdf.PdfReader") as mock_reader:
            mock_reader.return_value.pages = [mock_page1, mock_page2]
            text, page_count = extract_text_from_pdf(b"fake pdf bytes")
 
        assert page_count == 2
        assert "SECTION 1" in text
        assert "SECTION 2" in text
        print(f"    ✅ extract_text_from_pdf: {page_count} pages, {len(text)} chars")
 
    @patch("core.document_processor.generate_embedding")
    @patch("core.document_processor.classify_document")
    @patch("core.document_processor.pypdf.PdfReader")
    def test_process_document_full_pipeline(
        self, mock_reader, mock_classify, mock_embed
    ):
        """process_document should return a ProcessedDocument with correct fields."""
        # Mock PDF extraction
        mock_page = MagicMock()
        mock_page.extract_text.return_value = (
            "SECTION 7 HANDLING AND STORAGE. "
            "Store hydrochloric acid in ventilated areas away from incompatible materials. "
            "Use chemical-resistant containers. PPE: acid-resistant gloves and face shield required. "
            "Do not store near bases or reactive metals. Ensure proper grounding of containers. "
            "Emergency: flush with water immediately. Exposure limit: 5 ppm ceiling. "
            "UN 1789. Hazard Class 8. Packing Group II. Keep away from heat sources. "
        ) * 8  # Repeat to generate enough text for multiple chunks
        mock_reader.return_value.pages = [mock_page]
 
        # Mock LLM classification
        mock_classify.return_value = "sds"
 
        # Mock embeddings
        mock_embed.return_value = [0.1] * 3072
 
        from core.document_processor import process_document, ProcessedDocument
 
        result = process_document(
            pdf_bytes=b"fake pdf bytes",
            filename="hcl_sds.pdf",
            user_id="user_abc",
        )
 
        assert isinstance(result, ProcessedDocument)
        assert result.source == "hcl_sds.pdf"
        assert result.doc_type == "sds"
        assert result.total_chunks > 0
        assert result.total_pages == 1
        assert len(result.chunks) == result.total_chunks
 
        # Verify every chunk has the correct Azure index field names
        for chunk in result.chunks:
            assert "id"               in chunk, "Missing 'id'"
            assert "content"          in chunk, "Missing 'content'"
            assert "section"          in chunk, "Missing 'section'"
            assert "type"             in chunk, "Missing 'type'"
            assert "source"           in chunk, "Missing 'source'"
            assert "contentVector"    in chunk, "Missing 'contentVector'"
            assert "user_id"          in chunk, "Missing 'user_id'"
            assert "upload_timestamp" in chunk, "Missing 'upload_timestamp'"
 
            assert chunk["type"]          == "sds"
            assert chunk["source"]        == "hcl_sds.pdf"
            assert chunk["user_id"]       == "user_abc"
            assert len(chunk["contentVector"]) == 3072
 
        print(
            f"    ✅ process_document: produced {result.total_chunks} chunks "
            f"with all correct Azure index field names"
        )
 
    @patch("core.document_processor.generate_embedding")
    @patch("core.document_processor.classify_document")
    @patch("core.document_processor.pypdf.PdfReader")
    def test_process_document_chunk_ids_are_deterministic(
        self, mock_reader, mock_classify, mock_embed
    ):
        """Same filename + chunk index should always produce the same chunk ID."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hydrochloric acid safety data. " * 30
        mock_reader.return_value.pages = [mock_page]
        mock_classify.return_value = "sds"
        mock_embed.return_value = [0.0] * 3072
 
        from core.document_processor import process_document
 
        result1 = process_document(b"bytes", "test.pdf", "user_abc")
        result2 = process_document(b"bytes", "test.pdf", "user_abc")
 
        ids1 = [c["id"] for c in result1.chunks]
        ids2 = [c["id"] for c in result2.chunks]
        assert ids1 == ids2, "Chunk IDs must be deterministic for safe upserts"
        print("    ✅ process_document: chunk IDs are deterministic (safe for Azure upserts)")
 
 
# ══════════════════════════════════════════════════════════════
#  memory_manager Tests
# ══════════════════════════════════════════════════════════════
 
class TestMemoryManager(unittest.TestCase):
 
    def _fresh_session(self) -> dict:
        """Return a fresh empty session state dict."""
        return {}
 
    def test_init_memory_is_idempotent(self):
        """init_memory called multiple times should not corrupt state."""
        from core.memory_manager import init_memory
        session = self._fresh_session()
        init_memory(session)
        init_memory(session)
        init_memory(session)
        assert session["conversation_memory"] == []
        print("    ✅ init_memory: idempotent — safe to call multiple times")
 
    def test_add_and_get_turn(self):
        """add_turn should store user+assistant pair; get_memory should retrieve them."""
        from core.memory_manager import add_turn, get_memory
        session = self._fresh_session()
 
        add_turn(session, "What is the flash point?", "The flash point is 23°C.")
        memory = get_memory(session)
 
        assert len(memory) == 2
        assert memory[0]["role"]    == "user"
        assert memory[0]["content"] == "What is the flash point?"
        assert memory[1]["role"]    == "assistant"
        assert memory[1]["content"] == "The flash point is 23°C."
        print("    ✅ add_turn / get_memory: stores and retrieves user+assistant pair")
 
    def test_memory_trims_to_max_turns(self):
        """Memory should not grow beyond MAX_MEMORY_TURNS turns."""
        from core.memory_manager import add_turn, get_memory, get_turn_count
        from config.settings import MAX_MEMORY_TURNS
 
        session = self._fresh_session()
 
        # Add more turns than the limit
        for i in range(MAX_MEMORY_TURNS + 3):
            add_turn(session, f"Question {i}", f"Answer {i}")
 
        count = get_turn_count(session)
        assert count <= MAX_MEMORY_TURNS, (
            f"Expected <= {MAX_MEMORY_TURNS} turns, got {count}"
        )
        print(
            f"    ✅ memory trimming: capped at {count}/{MAX_MEMORY_TURNS} turns"
        )
 
    def test_format_memory_as_text_empty(self):
        """format_memory_as_text should return empty string when no memory."""
        from core.memory_manager import format_memory_as_text
        session = self._fresh_session()
        result = format_memory_as_text(session)
        assert result == ""
        print("    ✅ format_memory_as_text: returns '' for empty memory")
 
    def test_format_memory_as_text_with_turns(self):
        """format_memory_as_text should produce readable User/Assistant lines."""
        from core.memory_manager import add_turn, format_memory_as_text
        session = self._fresh_session()
        add_turn(session, "What PPE is needed?", "Acid-resistant gloves and face shield.")
 
        text = format_memory_as_text(session)
        assert "User:" in text
        assert "Assistant:" in text
        assert "What PPE is needed?" in text
        print(f"    ✅ format_memory_as_text:\n       {text}")
 
    def test_clear_memory(self):
        """clear_memory should wipe all turns."""
        from core.memory_manager import add_turn, clear_memory, get_turn_count
        session = self._fresh_session()
        add_turn(session, "Q1", "A1")
        add_turn(session, "Q2", "A2")
        assert get_turn_count(session) == 2
        clear_memory(session)
        assert get_turn_count(session) == 0
        print("    ✅ clear_memory: resets turn count to 0")
 
    def test_get_memory_returns_copy(self):
        """get_memory should return a list (not mutate session directly)."""
        from core.memory_manager import add_turn, get_memory
        session = self._fresh_session()
        add_turn(session, "Q", "A")
        mem = get_memory(session)
        mem.append({"role": "user", "content": "injected"})
        # Session memory should NOT have been mutated
        assert get_memory(session)[-1]["content"] != "injected"
        print("    ✅ get_memory: returns independent list (no session mutation)")
 
 
# ══════════════════════════════════════════════════════════════
#  tool_handler Tests
# ══════════════════════════════════════════════════════════════
 
class TestToolHandler(unittest.TestCase):
 
    def test_tool_schema_structure(self):
        """RETRIEVE_CHUNKS_TOOL must have correct OpenAI tool schema structure."""
        from core.tool_handler import RETRIEVE_CHUNKS_TOOL
 
        assert RETRIEVE_CHUNKS_TOOL["type"] == "function"
        fn = RETRIEVE_CHUNKS_TOOL["function"]
        assert fn["name"] == "retrieve_chunks"
        assert "description" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        props = params["properties"]
        assert "query"    in props
        assert "doc_type" in props
        assert "user_id"  in props
        assert set(params["required"]) == {"query", "doc_type", "user_id"}
        assert props["doc_type"]["enum"] == ["sds", "tds"]
        print("    ✅ RETRIEVE_CHUNKS_TOOL: schema is valid OpenAI tool definition")
 
    @patch("core.tool_handler.vector_search")
    @patch("core.tool_handler.generate_embedding")
    def test_execute_retrieve_chunks(self, mock_embed, mock_search):
        """execute_retrieve_chunks should embed query and call vector_search."""
        mock_embed.return_value = [0.1] * 3072
        mock_search.return_value = [
            {
                "id":               "chunk_001",
                "content":          "Store away from heat sources.",
                "section":          "Section 7",
                "type":             "sds",
                "source":           "hcl_sds.pdf",
                "upload_timestamp": "2026-03-04T10:00:00+00:00",
                "score":            0.91,
            }
        ]
 
        from core.tool_handler import execute_retrieve_chunks
        chunks = execute_retrieve_chunks(
            query="storage requirements for HCl",
            doc_type="sds",
            user_id="user_abc",
        )
 
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Store away from heat sources."
        mock_embed.assert_called_once_with("storage requirements for HCl")
        mock_search.assert_called_once()
        print("    ✅ execute_retrieve_chunks: embeds query and returns chunks")
 
    @patch("core.tool_handler.vector_search")
    @patch("core.tool_handler.generate_embedding")
    def test_dispatch_tool_call_returns_json(self, mock_embed, mock_search):
        """dispatch_tool_call should return valid JSON string."""
        mock_embed.return_value = [0.1] * 3072
        mock_search.return_value = [
            {
                "id": "c1", "content": "Wear gloves.", "section": "Sec 8",
                "type": "sds", "source": "hcl.pdf",
                "upload_timestamp": "2026-03-04T10:00:00+00:00", "score": 0.88,
            }
        ]
 
        from core.tool_handler import dispatch_tool_call
        result_json = dispatch_tool_call(
            tool_name="retrieve_chunks",
            tool_arguments_json=json.dumps({
                "query": "PPE requirements",
                "doc_type": "sds",
                "user_id": "user_abc",
            }),
            user_id="user_abc",
        )
 
        result = json.loads(result_json)
        assert "chunks" in result
        assert "total_retrieved" in result
        assert result["total_retrieved"] == 1
        assert result["chunks"][0]["content"] == "Wear gloves."
        assert "contentVector" not in result["chunks"][0], \
            "Raw vectors must NOT be returned to LLM"
        print("    ✅ dispatch_tool_call: returns correct JSON, excludes raw vectors")
 
    @patch("core.tool_handler.vector_search")
    @patch("core.tool_handler.generate_embedding")
    def test_dispatch_enforces_server_user_id(self, mock_embed, mock_search):
        """dispatch_tool_call must use server user_id, not LLM-provided one."""
        mock_embed.return_value = [0.0] * 3072
        mock_search.return_value = []
 
        from core.tool_handler import dispatch_tool_call
        dispatch_tool_call(
            tool_name="retrieve_chunks",
            tool_arguments_json=json.dumps({
                "query": "test",
                "doc_type": "sds",
                "user_id": "malicious_user_id",   # LLM tried to spoof
            }),
            user_id="real_user_abc",              # Server-side truth
        )
 
        # vector_search must have been called with the real user_id
        call_kwargs = mock_search.call_args
        actual_user_id = call_kwargs.kwargs.get("user_id") or call_kwargs.args[2]
        assert actual_user_id == "real_user_abc", \
            f"Server user_id not enforced! Got: {actual_user_id}"
        print("    ✅ dispatch_tool_call: server user_id enforced (injection prevention)")
 
    def test_dispatch_unknown_tool_raises(self):
        """dispatch_tool_call should raise ValueError for unknown tools."""
        from core.tool_handler import dispatch_tool_call
        with self.assertRaises(ValueError):
            dispatch_tool_call("unknown_tool", "{}", "user_abc")
        print("    ✅ dispatch_tool_call: raises ValueError for unknown tool name")
 
 
# ══════════════════════════════════════════════════════════════
#  query_engine Tests
# ══════════════════════════════════════════════════════════════
 
 
class TestQueryEngine(unittest.TestCase):
 
    def _make_text_response(self, content: str):
        """Build a mock LLM response that returns a plain text answer."""
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response
 
    def _make_tool_call_response(self, query: str, doc_type: str, user_id: str, call_id: str = "call_001"):
        """Build a mock LLM response that calls retrieve_chunks."""
        tool_call = MagicMock()
        tool_call.id = call_id
        tool_call.function.name = "retrieve_chunks"
        tool_call.function.arguments = json.dumps({
            "query": query,
            "doc_type": doc_type,
            "user_id": user_id,
        })
 
        msg = MagicMock()
        msg.tool_calls = [tool_call]
        msg.content = None
 
        choice = MagicMock()
        choice.message = msg
 
        response = MagicMock()
        response.choices = [choice]
        return response
 
    @patch("core.query_engine.classify_query")
    @patch("core.query_engine.chat_completion")
    def test_run_query_with_tool_call(self, mock_chat, mock_classify):
        """run_query should handle a full tool call cycle and return grounded answer."""
        mock_classify.return_value = "sds"
 
        # LLM first calls the tool, then answers
        mock_chat.side_effect = [
            self._make_tool_call_response("HCl storage", "sds", "user_abc"),
            self._make_text_response(
                "According to the SDS, store HCl in ventilated areas. *(Source: hcl_sds.pdf)*"
            ),
        ]
 
        with patch("core.query_engine.dispatch_tool_call") as mock_dispatch:
            mock_dispatch.return_value = json.dumps({
                "chunks": [{"content": "Store away from heat.", "source": "hcl_sds.pdf",
                            "section": "Section 7", "score": 0.92, "upload_timestamp": ""}],
                "total_retrieved": 1,
            })
 
            from core.query_engine import run_query
            result = run_query(
                user_question="How should I store hydrochloric acid?",
                user_id="user_abc",
                memory_messages=[],
                conversation_history_text="",
            )
 
        assert result.tool_was_called is True
        assert result.doc_type == "sds"
        assert result.iterations == 2
        assert len(result.chunks_retrieved) == 1
        assert "hcl_sds.pdf" in result.answer
        assert result.error is None
        print(
            f"    ✅ run_query (with tool call): "
            f"iterations={result.iterations}, "
            f"chunks={len(result.chunks_retrieved)}, "
            f"doc_type={result.doc_type}"
        )
 
    @patch("core.query_engine.classify_query")
    @patch("core.query_engine.chat_completion")
    def test_run_query_without_tool_call(self, mock_chat, mock_classify):
        """run_query should handle cases where LLM answers without retrieval (e.g. greetings)."""
        mock_classify.return_value = "sds"
        mock_chat.return_value = self._make_text_response("Hello! How can I help you today?")
 
        from core.query_engine import run_query
        result = run_query(
            user_question="Hello",
            user_id="user_abc",
            memory_messages=[],
            conversation_history_text="",
        )
 
        assert result.tool_was_called is False
        assert result.answer == "Hello! How can I help you today?"
        assert result.chunks_retrieved == []
        print("    ✅ run_query (no tool call): handles non-retrieval responses correctly")
 
    @patch("core.query_engine.classify_query")
    @patch("core.query_engine.chat_completion")
    def test_run_query_memory_injected(self, mock_chat, mock_classify):
        """run_query should inject memory messages into the LLM messages list."""
        mock_classify.return_value = "tds"
        mock_chat.return_value = self._make_text_response("The viscosity is 450 cP.")
 
        memory_messages = [
            {"role": "user",      "content": "What is product X?"},
            {"role": "assistant", "content": "Product X is an industrial lubricant."},
        ]
 
        from core.query_engine import run_query
        result = run_query(
            user_question="What is its viscosity?",
            user_id="user_abc",
            memory_messages=memory_messages,
            conversation_history_text="User: What is product X?\nAssistant: Product X is an industrial lubricant.",
        )
 
        # Inspect the messages passed to chat_completion
        call_args = mock_chat.call_args
        messages_sent = call_args.kwargs.get("messages") or call_args.args[0]
 
        roles = [m["role"] for m in messages_sent]
        assert "system"    in roles, "system prompt must be present"
        assert roles.count("user") >= 2, "memory user turn + current question must both be present"
        assert roles.count("assistant") >= 1, "memory assistant turn must be present"
        print(
            f"    ✅ run_query (memory injection): "
            f"messages={len(messages_sent)} "
            f"(system + {len(memory_messages)} memory + 1 current)"
        )
 
    @patch("core.query_engine.classify_query")
    @patch("core.query_engine.chat_completion")
    def test_run_query_tds_classification(self, mock_chat, mock_classify):
        """run_query should respect TDS classification for technical questions."""
        mock_classify.return_value = "tds"
        mock_chat.return_value = self._make_text_response("The cure temperature is 120°C.")
 
        from core.query_engine import run_query
        result = run_query(
            user_question="What is the recommended cure temperature?",
            user_id="user_abc",
            memory_messages=[],
            conversation_history_text="",
        )
 
        assert result.doc_type == "tds"
        print(f"    ✅ run_query: TDS classification propagated correctly")
 
 
# ══════════════════════════════════════════════════════════════
#  Main Runner
# ══════════════════════════════════════════════════════════════
 
def run_test_class(cls, label: str):
    print(f"\n[{label}] {cls.__name__}")
    loader = unittest.TestLoader()
    methods = sorted(m for m in dir(cls) if m.startswith("test_"))
    passed = failed = 0
    for method_name in methods:
        test = cls(method_name)
        try:
            test.debug()
            passed += 1
        except Exception as e:
            print(f"    ❌ {method_name}: {e}")
            failed += 1
    return passed, failed
 
 
def main():
    print("=" * 60)
    print("  Phase 3 Smoke Tests — Core Business Logic")
    print("=" * 60)
 
    total_passed = total_failed = 0
 
    for i, (cls, label) in enumerate([
        (TestDocumentProcessor, "1"),
        (TestMemoryManager,     "2"),
        (TestToolHandler,       "3"),
        (TestQueryEngine,       "4"),
    ], start=1):
        p, f = run_test_class(cls, label)
        total_passed += p
        total_failed += f
 
    print("\n" + "=" * 60)
    if total_failed == 0:
        print(f"  ✅ All {total_passed} Phase 3 tests passed!")
    else:
        print(f"  ❌ {total_failed} failed, {total_passed} passed.")
        sys.exit(1)
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()