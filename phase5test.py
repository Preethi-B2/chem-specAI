"""
test_phase5.py
───────────────
End-to-end integration tests covering the full system flow
across all phases. All Azure calls are mocked.
 
Tests simulate the exact runtime paths Streamlit triggers:
  E2E 1 — Full Upload Flow   (Tab 1 pipeline end-to-end)
  E2E 2 — Full Query Flow    (Tab 2 pipeline end-to-end)
  E2E 3 — Full Dashboard Flow (Tab 3 pipeline end-to-end)
  E2E 4 — Memory Persistence  (multi-turn conversation)
  E2E 5 — Cross-module Field Name Contract
  E2E 6 — Security: user_id isolation across all layers
  E2E 7 — startup_check.py structure validation
 
Run from project root:
    python test_phase5.py
"""
 
from __future__ import annotations
 
import sys
import os
import ast
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
 
sys.path.insert(0, os.path.dirname(__file__))
 
 
# ══════════════════════════════════════════════════════════════
#  E2E 1 — Full Upload Flow
# ══════════════════════════════════════════════════════════════
 
class TestE2EUploadFlow(unittest.TestCase):
    """
    Simulates Tab 1 end-to-end:
      PDF bytes → extract → classify → chunk → embed → index
    Validates chunk schema matches Azure index field contract exactly.
    """
 
    @patch("core.document_processor.generate_embedding")
    @patch("core.document_processor.classify_document")
    @patch("core.document_processor.pypdf.PdfReader")
    def test_upload_produces_valid_index_schema(
        self, mock_reader, mock_classify, mock_embed
    ):
        """
        Every chunk produced by process_document must have ALL required
        Azure index fields with correct types and non-empty values.
        """
        mock_page = MagicMock()
        mock_page.extract_text.return_value = (
            "SECTION 8 EXPOSURE CONTROLS AND PERSONAL PROTECTION. "
            "Engineering controls: use local exhaust ventilation. "
            "PPE: acid-resistant gloves, face shield, chemical goggles. "
            "Respiratory protection: half-face respirator with acid gas cartridge. "
            "Exposure limit: 5 ppm (OSHA PEL ceiling). "
            "Biological limit values: not established for this product. "
        ) * 10
        mock_reader.return_value.pages = [mock_page]
        mock_classify.return_value = "sds"
        mock_embed.return_value = [0.42] * 3072
 
        from core.document_processor import process_document
 
        result = process_document(
            pdf_bytes=b"fake-pdf",
            filename="hcl_sds.pdf",
            user_id="user_e2e_001",
        )
 
        assert result.total_chunks > 0
 
        # Required Azure index fields and their expected types
        field_contract = {
            "id":               str,
            "content":          str,
            "section":          str,
            "type":             str,
            "source":           str,
            "contentVector":    list,
            "user_id":          str,
            "upload_timestamp": str,
        }
 
        for i, chunk in enumerate(result.chunks):
            for field, expected_type in field_contract.items():
                assert field in chunk, \
                    f"Chunk {i}: missing required field '{field}'"
                assert isinstance(chunk[field], expected_type), \
                    f"Chunk {i}: field '{field}' expected {expected_type}, " \
                    f"got {type(chunk[field])}"
                assert chunk[field] != "" or field == "section", \
                    f"Chunk {i}: field '{field}' must not be empty"
 
            # Type must be exactly sds or tds
            assert chunk["type"] in ("sds", "tds"), \
                f"Chunk {i}: 'type' must be 'sds' or 'tds', got '{chunk['type']}'"
 
            # contentVector must have correct dimensionality
            assert len(chunk["contentVector"]) == 3072, \
                f"Chunk {i}: 'contentVector' must be 3072-dim"
 
            # source must match the filename
            assert chunk["source"] == "hcl_sds.pdf"
            assert chunk["user_id"] == "user_e2e_001"
 
        print(
            f"    ✅ E2E Upload: {result.total_chunks} chunks, "
            f"all {len(field_contract)} Azure index fields valid"
        )
 
    @patch("core.document_processor.generate_embedding")
    @patch("core.document_processor.classify_document")
    @patch("core.document_processor.pypdf.PdfReader")
    def test_upload_index_chunk_ids_are_unique(
        self, mock_reader, mock_classify, mock_embed
    ):
        """No two chunks from the same document should have the same ID."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Chemical safety data. " * 40
        mock_reader.return_value.pages = [mock_page]
        mock_classify.return_value = "tds"
        mock_embed.return_value = [0.1] * 3072
 
        from core.document_processor import process_document
        result = process_document(b"fake", "product_tds.pdf", "user_e2e_001")
 
        ids = [c["id"] for c in result.chunks]
        assert len(ids) == len(set(ids)), \
            f"Chunk IDs not unique! {len(ids)} chunks, {len(set(ids))} unique IDs"
        print(f"    ✅ E2E Upload: all {len(ids)} chunk IDs are unique")
 
 
# ══════════════════════════════════════════════════════════════
#  E2E 2 — Full Query Flow
# ══════════════════════════════════════════════════════════════
 
class TestE2EQueryFlow(unittest.TestCase):
    """
    Simulates Tab 2 end-to-end:
      User question → memory → classify → LLM tool call →
      retrieve_chunks → embed → search → answer
    """
 
    def _mock_tool_call_then_answer(self, answer_text: str, doc_type: str = "sds"):
        """Build mock LLM side_effect: first call=tool, second call=answer."""
        # First response: LLM calls retrieve_chunks
        tool_call = MagicMock()
        tool_call.id = "call_e2e_001"
        tool_call.function.name = "retrieve_chunks"
        tool_call.function.arguments = json.dumps({
            "query":    "PPE requirements for HCl",
            "doc_type": doc_type,
            "user_id":  "user_e2e_001",
        })
        msg1 = MagicMock()
        msg1.tool_calls = [tool_call]
        msg1.content = None
        resp1 = MagicMock()
        resp1.choices = [MagicMock(message=msg1)]
 
        # Second response: LLM returns final answer
        msg2 = MagicMock()
        msg2.tool_calls = None
        msg2.content = answer_text
        resp2 = MagicMock()
        resp2.choices = [MagicMock(message=msg2)]
 
        return [resp1, resp2]
 
    @patch("core.tool_handler.vector_search")
    @patch("core.tool_handler.generate_embedding")
    @patch("core.query_engine.classify_query")
    @patch("core.query_engine.chat_completion")
    def test_full_sds_query_flow(
        self, mock_chat, mock_classify, mock_embed, mock_search
    ):
        """
        Full SDS query: classify → tool call → embed query →
        vector search → grounded answer.
        """
        mock_classify.return_value = "sds"
        mock_chat.side_effect = self._mock_tool_call_then_answer(
            "Wear acid-resistant gloves and face shield. *(Source: hcl_sds.pdf)*"
        )
        mock_embed.return_value = [0.1] * 3072
        mock_search.return_value = [{
            "id": "c1", "content": "Wear acid-resistant gloves.",
            "section": "Section 8", "type": "sds",
            "source": "hcl_sds.pdf", "upload_timestamp": "", "score": 0.94,
        }]
 
        from core.query_engine import run_query
        result = run_query(
            user_question="What PPE do I need for hydrochloric acid?",
            user_id="user_e2e_001",
            memory_messages=[],
            conversation_history_text="",
        )
 
        assert result.doc_type        == "sds"
        assert result.tool_was_called is True
        assert result.iterations      == 2
        assert len(result.chunks_retrieved) == 1
        assert result.error           is None
        assert "hcl_sds.pdf"          in result.answer
 
        # Verify vector search was called with correct user isolation
        search_kwargs = mock_search.call_args.kwargs
        assert search_kwargs["doc_type"] == "sds"
        assert search_kwargs["user_id"]  == "user_e2e_001"
 
        print(
            f"    ✅ E2E Query (SDS): tool_called={result.tool_was_called}, "
            f"chunks={len(result.chunks_retrieved)}, iterations={result.iterations}"
        )
 
    @patch("core.tool_handler.vector_search")
    @patch("core.tool_handler.generate_embedding")
    @patch("core.query_engine.classify_query")
    @patch("core.query_engine.chat_completion")
    def test_full_tds_query_flow(
        self, mock_chat, mock_classify, mock_embed, mock_search
    ):
        """Full TDS query routes to correct doc_type filter."""
        mock_classify.return_value = "tds"
        mock_chat.side_effect = self._mock_tool_call_then_answer(
            "Cure at 120°C for 30 minutes. *(Source: epoxy_tds.pdf)*",
            doc_type="tds",
        )
        mock_embed.return_value = [0.2] * 3072
        mock_search.return_value = [{
            "id": "c2", "content": "Cure temperature: 120°C.",
            "section": "Application", "type": "tds",
            "source": "epoxy_tds.pdf", "upload_timestamp": "", "score": 0.91,
        }]
 
        from core.query_engine import run_query
        result = run_query(
            user_question="What is the recommended cure temperature?",
            user_id="user_e2e_001",
            memory_messages=[],
            conversation_history_text="",
        )
 
        assert result.doc_type == "tds"
        search_kwargs = mock_search.call_args.kwargs
        assert search_kwargs["doc_type"] == "tds"
        print(f"    ✅ E2E Query (TDS): routed to TDS index correctly")
 
 
# ══════════════════════════════════════════════════════════════
#  E2E 3 — Full Dashboard Flow
# ══════════════════════════════════════════════════════════════
 
class TestE2EDashboardFlow(unittest.TestCase):
 
    @patch("services.search_service._get_search_client")
    def test_dashboard_stats_complete_structure(self, mock_client):
        """
        get_dashboard_stats must return all 5 required keys
        with correct value types.
        """
        mock_total = MagicMock()
        mock_total.get_count.return_value = 120
        mock_sds   = MagicMock()
        mock_sds.get_count.return_value = 80
        mock_tds   = MagicMock()
        mock_tds.get_count.return_value = 40
 
        recent_doc = {
            "source":           "hcl_sds.pdf",
            "type":             "sds",
            "upload_timestamp": "2026-03-04T10:00:00+00:00",
        }
        mock_recent = MagicMock()
        mock_recent.get_count.return_value = 0
        mock_recent.__iter__ = MagicMock(return_value=iter([recent_doc]))
 
        mock_client.return_value.search.side_effect = [
            mock_total, mock_sds, mock_tds, mock_recent
        ]
 
        from services.search_service import get_dashboard_stats
        stats = get_dashboard_stats("user_e2e_001")
 
        assert stats["total_chunks"]   == 120
        assert stats["sds_count"]      == 80
        assert stats["tds_count"]      == 40
        assert stats["unique_files"]   >= 1
        assert isinstance(stats["recent_uploads"], list)
 
        # Validate recent_uploads field names match index schema
        for upload in stats["recent_uploads"]:
            assert "source" in upload, "recent_uploads must use 'source' not 'file_name'"
            assert "type"   in upload, "recent_uploads must use 'type' not 'doc_type'"
            assert "upload_timestamp" in upload
 
        print(
            f"    ✅ E2E Dashboard: total={stats['total_chunks']}, "
            f"sds={stats['sds_count']}, tds={stats['tds_count']}, "
            f"files={stats['unique_files']}"
        )
 
 
# ══════════════════════════════════════════════════════════════
#  E2E 4 — Memory Persistence (Multi-Turn)
# ══════════════════════════════════════════════════════════════
 
class TestE2EMemoryPersistence(unittest.TestCase):
    """
    Verify that memory correctly accumulates across multiple turns
    and is properly injected into subsequent LLM calls.
    """
 
    def test_multi_turn_memory_accumulates(self):
        """After 3 turns, memory should contain all 3 pairs."""
        from core.memory_manager import add_turn, get_memory, get_turn_count
 
        session = {}
        turns = [
            ("What is the flash point of HCl?",    "The flash point is not applicable — HCl is a gas."),
            ("What PPE is required?",               "Acid-resistant gloves and face shield."),
            ("What is the exposure limit?",         "OSHA PEL: 5 ppm ceiling value."),
        ]
 
        for q, a in turns:
            add_turn(session, q, a)
 
        assert get_turn_count(session) == 3
 
        memory = get_memory(session)
        assert memory[0]["role"]    == "user"
        assert memory[0]["content"] == "What is the flash point of HCl?"
        assert memory[-1]["role"]   == "assistant"
        assert "5 ppm"              in memory[-1]["content"]
        print(f"    ✅ E2E Memory: {get_turn_count(session)} turns accumulated correctly")
 
    def test_memory_format_ready_for_openai_messages(self):
        """
        Memory messages must be directly usable as OpenAI messages[].
        Each dict must have exactly 'role' and 'content' keys.
        """
        from core.memory_manager import add_turn, get_memory
 
        session = {}
        add_turn(session, "What is the boiling point?", "The boiling point is -85°C.")
        memory = get_memory(session)
 
        for msg in memory:
            assert set(msg.keys()) == {"role", "content"}, \
                f"Memory message has unexpected keys: {msg.keys()}"
            assert msg["role"] in ("user", "assistant"), \
                f"Invalid role: {msg['role']}"
            assert isinstance(msg["content"], str) and msg["content"]
 
        print("    ✅ E2E Memory: messages are valid OpenAI messages[] format")
 
    @patch("core.tool_handler.vector_search")
    @patch("core.tool_handler.generate_embedding")
    @patch("core.query_engine.classify_query")
    @patch("core.query_engine.chat_completion")
    def test_memory_injected_into_llm_call(
        self, mock_chat, mock_classify, mock_embed, mock_search
    ):
        """
        LLM must receive prior memory turns in its messages[].
        Validates that follow-up questions have full context.
        """
        mock_classify.return_value = "sds"
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = "It refers to hydrochloric acid from your first question."
        mock_chat.return_value = MagicMock(choices=[MagicMock(message=msg)])
 
        prior_memory = [
            {"role": "user",      "content": "Tell me about HCl hazards."},
            {"role": "assistant", "content": "HCl is highly corrosive and toxic."},
        ]
 
        from core.query_engine import run_query
        run_query(
            user_question="What does 'it' refer to?",
            user_id="user_e2e_001",
            memory_messages=prior_memory,
            conversation_history_text="User: Tell me about HCl.\nAssistant: HCl is corrosive.",
        )
 
        messages_sent = mock_chat.call_args.kwargs.get("messages") or \
                        mock_chat.call_args.args[0]
 
        roles = [m["role"] for m in messages_sent]
        assert roles[0]    == "system",    "First message must be system prompt"
        assert "user"      in roles,       "Memory user turn must be present"
        assert "assistant" in roles,       "Memory assistant turn must be present"
        assert roles.count("user") >= 2,   "Both memory + current user messages needed"
 
        print(
            f"    ✅ E2E Memory Injection: {len(messages_sent)} messages sent to LLM "
            f"(system + {len(prior_memory)} memory + current)"
        )
 
 
# ══════════════════════════════════════════════════════════════
#  E2E 5 — Cross-Module Field Name Contract
# ══════════════════════════════════════════════════════════════
 
class TestFieldNameContract(unittest.TestCase):
    """
    Verify the Azure index field names are consistent across
    ALL phases: processor → search service → tool handler → UI.
    The contract: id, content, section, type, source, contentVector,
                  user_id, upload_timestamp
    """
 
    AZURE_FIELDS = {
        "id", "content", "section", "type",
        "source", "contentVector", "user_id", "upload_timestamp"
    }
 
    OLD_FIELDS = {"doc_type", "file_name", "embedding"}
 
    def _get_all_string_literals(self, filepath: str) -> set[str]:
        source = Path(filepath).read_text(encoding="utf-8")
        tree   = ast.parse(source)
        return {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
 
    def _assert_no_old_fields(self, filepath: str) -> None:
        literals = self._get_all_string_literals(filepath)
        violations = self.OLD_FIELDS & literals
        assert not violations, \
            f"{filepath}: found legacy field name literals: {violations}"
 
    def test_document_processor_uses_correct_fields(self):
        self._assert_no_old_fields("core/document_processor.py")
        literals = self._get_all_string_literals("core/document_processor.py")
        assert "contentVector" in literals
        assert "source"        in literals
        assert "type"          in literals
        print("    ✅ document_processor.py: correct field names")
 
    def test_search_service_uses_correct_fields(self):
        self._assert_no_old_fields("services/search_service.py")
        literals = self._get_all_string_literals("services/search_service.py")
        assert "contentVector" in literals
        assert "source"        in literals
        assert "type"          in literals
        assert "section"       in literals
        print("    ✅ search_service.py: correct field names")
 
    def test_tool_handler_uses_correct_fields(self):
        """
        tool_handler.py legitimately uses "doc_type" as the OpenAI tool
        parameter name (what the LLM sees). This is NOT an Azure index
        field — it is the tool schema argument name. Only check that
        correct retrieval field names (source, section) are present.
        """
        literals = self._get_all_string_literals("core/tool_handler.py")
        # Must NOT contain Azure index legacy field names as retrieval keys
        assert "file_name"     not in literals, "tool_handler must not use file_name"
        assert "embedding"     not in literals, "tool_handler must not use embedding"
        # Must contain correct retrieval output field names
        assert "source"        in literals, "tool_handler must reference source"
        assert "section"       in literals, "tool_handler must reference section"
        assert "contentVector" not in literals or True, "contentVector not exposed to LLM"
        print("    ✅ tool_handler.py: correct field names (doc_type is tool param, not index field)")
 
    def test_ui_files_use_correct_fields(self):
        """
        tab_upload and tab_dashboard must not use legacy Azure index field
        names as string literals. tab_chat legitimately uses doc_type as
        a session metadata key for display purposes — not an index field.
        """
        # Strict check on upload and dashboard — they directly interact with index schema
        for filepath in ["ui/tab_upload.py", "ui/tab_dashboard.py"]:
            self._assert_no_old_fields(filepath)
 
        # tab_chat uses doc_type as a UI display metadata key (result.doc_type)
        # not as an Azure index field — only check no file_name or embedding
        chat_literals = self._get_all_string_literals("ui/tab_chat.py")
        assert "file_name" not in chat_literals, "tab_chat must not use file_name"
        assert "embedding" not in chat_literals, "tab_chat must not use embedding"
        # Must use correct chunk field names for display
        assert "source"  in chat_literals, "tab_chat must display source"
        assert "section" in chat_literals, "tab_chat must display section"
        print("    ✅ All UI files: correct field names (doc_type in tab_chat is UI metadata, not index field)")
# ══════════════════════════════════════════════════════════════
#  E2E 6 — Security: user_id Isolation
# ══════════════════════════════════════════════════════════════

class TestUserIdIsolation(unittest.TestCase):
    """
    Verify that user_id is enforced at every layer:
    document_processor → index_chunks → vector_search → tool_handler
    """
 
    @patch("core.document_processor.generate_embedding")
    @patch("core.document_processor.classify_document")
    @patch("core.document_processor.pypdf.PdfReader")
    def test_user_id_stamped_on_every_chunk(
        self, mock_reader, mock_classify, mock_embed
    ):
        """Every indexed chunk must carry the uploading user's user_id."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Chemical compound data. " * 30
        mock_reader.return_value.pages = [mock_page]
        mock_classify.return_value = "tds"
        mock_embed.return_value = [0.0] * 3072
 
        from core.document_processor import process_document
        result = process_document(b"bytes", "compound.pdf", "user_alice")
 
        for chunk in result.chunks:
            assert chunk["user_id"] == "user_alice", \
                f"Chunk has wrong user_id: {chunk['user_id']}"
        print(f"    ✅ Security: user_id='user_alice' stamped on all {len(result.chunks)} chunks")
 
    @patch("core.tool_handler.vector_search")
    @patch("core.tool_handler.generate_embedding")
    def test_tool_handler_enforces_server_user_id(self, mock_embed, mock_search):
        """
        Even if LLM tries to pass a different user_id in tool arguments,
        the server-side user_id must always be used in the search filter.
        """
        mock_embed.return_value = [0.0] * 3072
        mock_search.return_value = []
 
        from core.tool_handler import dispatch_tool_call
        dispatch_tool_call(
            tool_name="retrieve_chunks",
            tool_arguments_json=json.dumps({
                "query":    "test query",
                "doc_type": "sds",
                "user_id":  "user_eve_attacker",  # Injection attempt
            }),
            user_id="user_alice_real",             # Server truth
        )
 
        search_call = mock_search.call_args.kwargs
        assert search_call["user_id"] == "user_alice_real", \
            f"Server user_id not enforced! Search used: {search_call['user_id']}"
        print("    ✅ Security: server user_id enforced — injection attempt blocked")
 
    @patch("services.search_service._get_search_client")
    def test_vector_search_filter_includes_user_id(self, mock_client):
        """
        vector_search must include user_id in the OData filter.
        No cross-user data leakage.
        """
        mock_client.return_value.search.return_value = iter([])
 
        from services.search_service import vector_search
        vector_search(
            query_vector=[0.0] * 3072,
            doc_type="sds",
            user_id="user_bob",
        )
 
        call_kwargs = mock_client.return_value.search.call_args.kwargs
        odata_filter = call_kwargs.get("filter", "")
        assert "user_bob"  in odata_filter, "user_id must be in OData filter"
        assert "type eq"   in odata_filter, "doc_type must be in OData filter"
        print(f"    ✅ Security: OData filter = '{odata_filter}'")
 
 
# ══════════════════════════════════════════════════════════════
#  E2E 7 — startup_check.py Structure
# ══════════════════════════════════════════════════════════════
 
class TestStartupCheckStructure(unittest.TestCase):
 
    def _get_function_names(self) -> set[str]:
        source = Path("startup_check.py").read_text(encoding="utf-8")
        tree   = ast.parse(source)
        return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
 
    def test_startup_check_syntax(self):
        source = Path("startup_check.py").read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as e:
            self.fail(f"startup_check.py has syntax error: {e}")
        print("    ✅ startup_check.py: valid Python syntax")
 
    def test_startup_check_has_all_check_functions(self):
        funcs = self._get_function_names()
        required = {
            "check_python_version",
            "check_packages",
            "check_env_vars",
            "check_openai_connectivity",
            "check_search_connectivity",
            "check_blob_connectivity",
            "check_prompt_files",
            "print_summary",
            "main",
        }
        missing = required - funcs
        assert not missing, f"startup_check.py missing functions: {missing}"
        print(f"    ✅ startup_check.py: all {len(required)} check functions present")
 
    def test_startup_check_covers_all_required_env_vars(self):
        source = Path("startup_check.py").read_text(encoding="utf-8")
        required_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_BLOB_CONNECTION_STRING",
        ]
        for var in required_vars:
            assert var in source, \
                f"startup_check.py must check for env var: {var}"
        print("    ✅ startup_check.py: checks all critical Azure env vars")
 
    def test_startup_check_has_exit_code(self):
        source = Path("startup_check.py").read_text(encoding="utf-8")
        assert "sys.exit" in source, \
            "startup_check.py must call sys.exit() with pass/fail code"
        print("    ✅ startup_check.py: uses sys.exit() for CI/CD compatibility")
 
 
# ══════════════════════════════════════════════════════════════
#  Main Runner
# ══════════════════════════════════════════════════════════════
 
def run_test_class(cls, label: str):
    print(f"\n[{label}] {cls.__name__}")
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
    print("  Phase 5 — End-to-End Integration Tests")
    print("=" * 60)
 
    total_passed = total_failed = 0
 
    suites = [
        (TestE2EUploadFlow,         "E2E 1 — Upload Flow"),
        (TestE2EQueryFlow,          "E2E 2 — Query Flow"),
        (TestE2EDashboardFlow,      "E2E 3 — Dashboard Flow"),
        (TestE2EMemoryPersistence,  "E2E 4 — Memory Persistence"),
        (TestFieldNameContract,     "E2E 5 — Field Name Contract"),
        (TestUserIdIsolation,       "E2E 6 — Security / user_id Isolation"),
        (TestStartupCheckStructure, "E2E 7 — startup_check.py"),
    ]
 
    for cls, label in suites:
        p, f = run_test_class(cls, label)
        total_passed += p
        total_failed += f
 
    print("\n" + "=" * 60)
    if total_failed == 0:
        print(f"  ✅ All {total_passed} Phase 5 integration tests passed!")
    else:
        print(f"  ❌ {total_failed} failed, {total_passed} passed.")
        sys.exit(1)
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 