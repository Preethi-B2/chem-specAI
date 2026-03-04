"""
test_phase4.py
───────────────
Structural and wiring tests for Phase 4 UI files.
Validates imports, function signatures, and inter-module wiring
without running a live Streamlit server.
 
Run from project root:
    python test_phase4.py
"""
 
from __future__ import annotations
 
import sys
import os
import ast
import importlib
import unittest
from pathlib import Path
 
sys.path.insert(0, os.path.dirname(__file__))
 
 
# ══════════════════════════════════════════════════════════════
#  Syntax & Import Validation
# ══════════════════════════════════════════════════════════════
 
class TestFileSyntax(unittest.TestCase):
    """All Phase 4 files must be valid Python with no syntax errors."""
 
    def _assert_valid_syntax(self, filepath: str) -> None:
        source = Path(filepath).read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as e:
            self.fail(f"Syntax error in '{filepath}': {e}")
 
    def test_app_py_syntax(self):
        self._assert_valid_syntax("app.py")
        print("    ✅ app.py: valid Python syntax")
 
    def test_tab_upload_syntax(self):
        self._assert_valid_syntax("ui/tab_upload.py")
        print("    ✅ ui/tab_upload.py: valid Python syntax")
 
    def test_tab_chat_syntax(self):
        self._assert_valid_syntax("ui/tab_chat.py")
        print("    ✅ ui/tab_chat.py: valid Python syntax")
 
    def test_tab_dashboard_syntax(self):
        self._assert_valid_syntax("ui/tab_dashboard.py")
        print("    ✅ ui/tab_dashboard.py: valid Python syntax")
 
 
# ══════════════════════════════════════════════════════════════
#  Import Wiring Tests
# ══════════════════════════════════════════════════════════════
 
class TestImportWiring(unittest.TestCase):
    """
    Validate that each UI module imports from the correct Phase 1/2/3 modules.
    Parses AST import nodes — does not execute any code.
    """
 
    def _get_imports(self, filepath: str) -> set[str]:
        """Return all module names imported in a file."""
        source = Path(filepath).read_text(encoding="utf-8")
        tree   = ast.parse(source)
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        return imports
 
    def test_app_imports_all_tabs(self):
        imports = self._get_imports("app.py")
        assert "ui.tab_upload"    in imports, "app.py must import ui.tab_upload"
        assert "ui.tab_chat"      in imports, "app.py must import ui.tab_chat"
        assert "ui.tab_dashboard" in imports, "app.py must import ui.tab_dashboard"
        print("    ✅ app.py: imports all 3 tab modules")
 
    def test_app_does_not_import_azure_directly(self):
        """app.py must not call Azure services directly — only tab modules."""
        imports = self._get_imports("app.py")
        forbidden = {"services.openai_service", "services.blob_service", "services.search_service"}
        violations = forbidden & imports
        assert not violations, (
            f"app.py must not import Azure services directly: {violations}"
        )
        print("    ✅ app.py: no direct Azure service imports (clean routing layer)")
 
    def test_tab_upload_imports_correct_modules(self):
        imports = self._get_imports("ui/tab_upload.py")
        assert "core.document_processor"   in imports, "tab_upload must use document_processor"
        assert "services.blob_service"     in imports, "tab_upload must use blob_service"
        assert "services.search_service"   in imports, "tab_upload must use search_service"
        assert "utils.helpers"             in imports, "tab_upload must use helpers"
        print("    ✅ ui/tab_upload.py: correct module imports")
 
    def test_tab_upload_does_not_import_openai_directly(self):
        """tab_upload must not call OpenAI directly — only via document_processor."""
        imports = self._get_imports("ui/tab_upload.py")
        assert "services.openai_service" not in imports, (
            "tab_upload must not call OpenAI directly. Use document_processor instead."
        )
        print("    ✅ ui/tab_upload.py: no direct OpenAI calls (uses document_processor)")
 
    def test_tab_chat_imports_correct_modules(self):
        imports = self._get_imports("ui/tab_chat.py")
        assert "core.memory_manager" in imports, "tab_chat must use memory_manager"
        assert "core.query_engine"   in imports, "tab_chat must use query_engine"
        assert "utils.helpers"       in imports, "tab_chat must use helpers"
        print("    ✅ ui/tab_chat.py: correct module imports")
 
    def test_tab_chat_does_not_import_search_directly(self):
        """tab_chat must not call search_service directly — only via query_engine."""
        imports = self._get_imports("ui/tab_chat.py")
        assert "services.search_service" not in imports, (
            "tab_chat must not call search_service directly. Use query_engine instead."
        )
        print("    ✅ ui/tab_chat.py: no direct search calls (uses query_engine)")
 
    def test_tab_dashboard_imports_search_service(self):
        imports = self._get_imports("ui/tab_dashboard.py")
        assert "services.search_service" in imports, \
            "tab_dashboard must use search_service for analytics"
        print("    ✅ ui/tab_dashboard.py: imports search_service for live analytics")
 
 
# ══════════════════════════════════════════════════════════════
#  Function Signature Tests
# ══════════════════════════════════════════════════════════════
 
class TestFunctionSignatures(unittest.TestCase):
    """
    Confirm that each UI module exposes the expected render function
    with the correct name — these are called directly from app.py.
    """
 
    def _get_function_names(self, filepath: str) -> set[str]:
        source = Path(filepath).read_text(encoding="utf-8")
        tree   = ast.parse(source)
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
 
    def test_app_has_main(self):
        funcs = self._get_function_names("app.py")
        assert "main" in funcs, "app.py must define main()"
        print("    ✅ app.py: defines main()")
 
    def test_tab_upload_has_render_function(self):
        funcs = self._get_function_names("ui/tab_upload.py")
        assert "render_upload_tab" in funcs, \
            "tab_upload.py must define render_upload_tab()"
        print("    ✅ ui/tab_upload.py: defines render_upload_tab()")
 
    def test_tab_chat_has_render_function(self):
        funcs = self._get_function_names("ui/tab_chat.py")
        assert "render_chat_tab" in funcs, \
            "tab_chat.py must define render_chat_tab()"
        print("    ✅ ui/tab_chat.py: defines render_chat_tab()")
 
    def test_tab_dashboard_has_render_function(self):
        funcs = self._get_function_names("ui/tab_dashboard.py")
        assert "render_dashboard_tab" in funcs, \
            "tab_dashboard.py must define render_dashboard_tab()"
        print("    ✅ ui/tab_dashboard.py: defines render_dashboard_tab()")
 
    def test_tab_upload_has_pipeline_function(self):
        funcs = self._get_function_names("ui/tab_upload.py")
        assert "_run_upload_pipeline" in funcs, \
            "tab_upload.py must define _run_upload_pipeline()"
        print("    ✅ ui/tab_upload.py: defines _run_upload_pipeline()")
 
    def test_tab_chat_has_message_handler(self):
        funcs = self._get_function_names("ui/tab_chat.py")
        assert "_handle_user_message" in funcs, \
            "tab_chat.py must define _handle_user_message()"
        print("    ✅ ui/tab_chat.py: defines _handle_user_message()")
 
    def test_tab_dashboard_has_kpi_and_table(self):
        funcs = self._get_function_names("ui/tab_dashboard.py")
        assert "_render_kpi_metrics"          in funcs
        assert "_render_recent_uploads_table" in funcs
        assert "_render_distribution_chart"   in funcs
        print("    ✅ ui/tab_dashboard.py: defines all rendering sub-functions")
 
 
# ══════════════════════════════════════════════════════════════
#  Field Name Consistency Tests
# ══════════════════════════════════════════════════════════════
 
class TestFieldNameConsistency(unittest.TestCase):
    """
    Ensure UI files reference the correct Azure index field names
    as string literals: 'source', 'type', 'section', 'contentVector'.
    Checks string literals only (AST) — avoids false positives from
    variable names and object attributes like processed.doc_type.
    """
 
    def _get_string_literals(self, filepath: str) -> set:
        """Return all string literal values in a Python file via AST parse."""
        source = Path(filepath).read_text(encoding="utf-8")
        tree   = ast.parse(source)
        return {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
 
    def test_tab_upload_no_old_field_name_literals(self):
        literals = self._get_string_literals("ui/tab_upload.py")
        assert "doc_type"  not in literals, \
            "tab_upload.py must not use 'doc_type' as a string literal (use 'type')"
        assert "file_name" not in literals, \
            "tab_upload.py must not use 'file_name' as a string literal (use 'source')"
        assert "embedding" not in literals, \
            "tab_upload.py must not use 'embedding' as a string literal (use 'contentVector')"
        print("    ✅ ui/tab_upload.py: no legacy field name string literals")
 
    def test_tab_chat_uses_correct_chunk_fields(self):
        literals = self._get_string_literals("ui/tab_chat.py")
        assert "source"  in literals, \
            "tab_chat.py must use 'source' as a dict key string"
        assert "section" in literals, \
            "tab_chat.py must use 'section' as a dict key string"
        print("    ✅ ui/tab_chat.py: uses correct chunk field name strings (source, section)")
 
    def test_tab_dashboard_uses_correct_field_name_literals(self):
        literals = self._get_string_literals("ui/tab_dashboard.py")
        assert "file_name" not in literals, \
            "tab_dashboard.py must not use 'file_name' as a string literal (use 'source')"
        assert "source" in literals, \
            "tab_dashboard.py must use 'source' as a string key"
        assert "type"   in literals, \
            "tab_dashboard.py must use 'type' as a string key"
        print("    ✅ ui/tab_dashboard.py: correct field name string literals (source, type)")
 
 

 
#  app.py Structure Tests
# ══════════════════════════════════════════════════════════════
 
class TestAppStructure(unittest.TestCase):
    """Validate app.py structure — tabs, page config, session init."""
 
    def _read(self) -> str:
        return Path("app.py").read_text(encoding="utf-8")
 
    def test_app_has_set_page_config(self):
        source = self._read()
        assert "set_page_config" in source, "app.py must call st.set_page_config()"
        print("    ✅ app.py: calls st.set_page_config()")
 
    def test_app_has_three_tabs(self):
        source = self._read()
        assert "st.tabs" in source, "app.py must use st.tabs()"
        assert "Upload"    in source
        assert "Chat"      in source
        assert "Dashboard" in source
        print("    ✅ app.py: defines all 3 tabs (Upload, Chat, Dashboard)")
 
    def test_app_calls_all_render_functions(self):
        source = self._read()
        assert "render_upload_tab()"    in source
        assert "render_chat_tab()"      in source
        assert "render_dashboard_tab()" in source
        print("    ✅ app.py: calls all 3 render functions")
 
    def test_app_initialises_session_state(self):
        source = self._read()
        assert "_init_session" in source or "session_state" in source
        print("    ✅ app.py: initialises session state")
 
    def test_app_has_entrypoint_guard(self):
        source = self._read()
        assert 'if __name__ == "__main__"' in source or \
               "if __name__ == '__main__'" in source
        print("    ✅ app.py: has __main__ entrypoint guard")
 
 
# ══════════════════════════════════════════════════════════════
#  Main Runner
# ══════════════════════════════════════════════════════════════
 
def run_test_class(cls, label: str):
    print(f"\n[{label}] {cls.__name__}")
    loader  = unittest.TestLoader()
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
    print("  Phase 4 Tests — Streamlit UI Wiring & Structure")
    print("=" * 60)
 
    total_passed = total_failed = 0
 
    for i, (cls, label) in enumerate([
        (TestFileSyntax,           "1"),
        (TestImportWiring,         "2"),
        (TestFunctionSignatures,   "3"),
        (TestFieldNameConsistency, "4"),
        (TestAppStructure,         "5"),
    ], start=1):
        p, f = run_test_class(cls, label)
        total_passed += p
        total_failed += f
 
    print("\n" + "=" * 60)
    if total_failed == 0:
        print(f"  ✅ All {total_passed} Phase 4 tests passed!")
    else:
        print(f"  ❌ {total_failed} failed, {total_passed} passed.")
        sys.exit(1)
    print("=" * 60)
    print("\n  ▶ To run the app:  streamlit run app.py")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 