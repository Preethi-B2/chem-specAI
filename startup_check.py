"""
startup_check.py
─────────────────
Pre-flight validator that checks all required configuration and Azure
connectivity BEFORE launching the Streamlit app.
 
Run manually:
    python startup_check.py
 
Or integrate into a launch script:
    python startup_check.py && streamlit run app.py
 
Exit codes:
    0 — All checks passed, safe to launch
    1 — One or more checks failed, do not launch
 
Checks performed:
    1. Python version >= 3.10
    2. All required packages installed
    3. All required .env variables present and non-empty
    4. Azure OpenAI reachable (test embedding call)
    5. Azure AI Search reachable (index check)
    6. Azure Blob Storage reachable (container check)
    7. All prompt .md files present and non-empty
"""
 
from __future__ import annotations
 
import sys
import importlib
 
# ── ANSI colours ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
 
PASS = f"{GREEN}  ✅ PASS{RESET}"
FAIL = f"{RED}  ❌ FAIL{RESET}"
WARN = f"{YELLOW}  ⚠️  WARN{RESET}"
 
_failures: list[str] = []
 
 
def _ok(msg: str) -> None:
    print(f"{PASS}  {msg}")
 
 
def _fail(msg: str) -> None:
    print(f"{FAIL}  {msg}")
    _failures.append(msg)
 
 
def _warn(msg: str) -> None:
    print(f"{WARN}  {msg}")
 
 
def _section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 55}{RESET}")
 
 
# ══════════════════════════════════════════════════════════════
#  Check 1 — Python Version
# ══════════════════════════════════════════════════════════════
 
def check_python_version() -> None:
    _section("1 / 7 — Python Version")
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major == 3 and minor >= 10:
        _ok(f"Python {version_str} (>= 3.10 required)")
    else:
        _fail(f"Python {version_str} — requires 3.10 or higher")
 
 
# ══════════════════════════════════════════════════════════════
#  Check 2 — Required Packages
# ══════════════════════════════════════════════════════════════
 
_REQUIRED_PACKAGES = {
    "streamlit":                    "streamlit",
    "openai":                       "openai",
    "azure.search.documents":       "azure-search-documents",
    "azure.storage.blob":           "azure-storage-blob",
    "pypdf":                        "pypdf",
    "dotenv":                       "python-dotenv",
    "tiktoken":                     "tiktoken",
    "tenacity":                     "tenacity",
}
 
 
def check_packages() -> None:
    _section("2 / 7 — Required Packages")
    for module_name, pip_name in _REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            _ok(f"{pip_name}")
        except ImportError:
            _fail(f"{pip_name} not installed — run: pip install {pip_name}")
 
 
# ══════════════════════════════════════════════════════════════
#  Check 3 — Environment Variables
# ══════════════════════════════════════════════════════════════
 
_REQUIRED_ENV_VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX_NAME",
    "AZURE_BLOB_CONNECTION_STRING",
    "AZURE_BLOB_CONTAINER_NAME",
]
 
 
def check_env_vars() -> None:
    _section("3 / 7 — Environment Variables (.env)")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        _ok(".env file loaded")
    except Exception as e:
        _fail(f"Failed to load .env: {e}")
        return
 
    import os
    for var in _REQUIRED_ENV_VARS:
        value = os.getenv(var, "")
        if value and not value.startswith("<"):
            _ok(f"{var}")
        elif value.startswith("<"):
            _fail(f"{var} — still has placeholder value: '{value[:40]}'")
        else:
            _fail(f"{var} — not set or empty")
 
 
# ══════════════════════════════════════════════════════════════
#  Check 4 — Azure OpenAI Connectivity
# ══════════════════════════════════════════════════════════════
 
def check_openai_connectivity() -> None:
    _section("4 / 7 — Azure OpenAI Connectivity")
    try:
        from services.openai_service import generate_embedding
        vector = generate_embedding("startup connectivity test")
        if isinstance(vector, list) and len(vector) > 0:
            _ok(f"Embedding model reachable — vector dim: {len(vector)}")
        else:
            _fail("Embedding returned empty or invalid response")
    except Exception as e:
        _fail(f"Azure OpenAI unreachable: {e}")
 
 
# ══════════════════════════════════════════════════════════════
#  Check 5 — Azure AI Search Connectivity
# ══════════════════════════════════════════════════════════════
 
def check_search_connectivity() -> None:
    _section("5 / 7 — Azure AI Search Connectivity")
    try:
        from services.search_service import ensure_index_exists
        ensure_index_exists()
        _ok("Azure AI Search reachable — index verified/created")
    except Exception as e:
        _fail(f"Azure AI Search unreachable: {e}")
 
 
# ══════════════════════════════════════════════════════════════
#  Check 6 — Azure Blob Storage Connectivity
# ══════════════════════════════════════════════════════════════
 
def check_blob_connectivity() -> None:
    _section("6 / 7 — Azure Blob Storage Connectivity")
    try:
        from services.blob_service import ensure_container_exists
        ensure_container_exists()
        _ok("Azure Blob Storage reachable — container verified/created")
    except Exception as e:
        _fail(f"Azure Blob Storage unreachable: {e}")
 
 
# ══════════════════════════════════════════════════════════════
#  Check 7 — Prompt Files
# ══════════════════════════════════════════════════════════════
 
_REQUIRED_PROMPTS = [
    "system_prompt.md",
    "document_classifier.md",
    "query_classifier.md",
    "answer_generator.md",
]
 
 
def check_prompt_files() -> None:
    _section("7 / 7 — Prompt Files (prompts/*.md)")
    try:
        from utils.prompt_loader import load_prompt, list_available_prompts
        available = list_available_prompts()
 
        for fname in _REQUIRED_PROMPTS:
            if fname not in available:
                _fail(f"prompts/{fname} — file not found")
                continue
            content = load_prompt(fname)
            if len(content) < 50:
                _warn(f"prompts/{fname} — suspiciously short ({len(content)} chars)")
            else:
                _ok(f"prompts/{fname} ({len(content)} chars)")
    except Exception as e:
        _fail(f"Prompt loader error: {e}")
 
 
# ══════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════
 
def print_summary() -> int:
    print(f"\n{BOLD}{'═' * 55}{RESET}")
    if not _failures:
        print(f"{GREEN}{BOLD}  ✅ ALL CHECKS PASSED — Safe to launch{RESET}")
        print(f"{BOLD}{'═' * 55}{RESET}")
        print(f"\n  ▶  streamlit run app.py\n")
        return 0
    else:
        print(f"{RED}{BOLD}  ❌ {len(_failures)} CHECK(S) FAILED — Fix before launching{RESET}")
        print(f"{BOLD}{'═' * 55}{RESET}")
        print(f"\n{RED}  Failed checks:{RESET}")
        for i, msg in enumerate(_failures, 1):
            print(f"  {i}. {msg}")
        print()
        return 1
 
 
# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
 
def main() -> None:
    print(f"\n{BOLD}{'═' * 55}{RESET}")
    print(f"{BOLD}  Chemistry Doc Intelligence — Pre-flight Check{RESET}")
    print(f"{BOLD}{'═' * 55}{RESET}")
 
    check_python_version()
    check_packages()
    check_env_vars()
    check_prompt_files()
 
    # Azure connectivity checks only run if env vars are set
    if not _failures or all("not set" not in f and "placeholder" not in f for f in _failures):
        check_openai_connectivity()
        check_search_connectivity()
        check_blob_connectivity()
    else:
        print(f"\n{YELLOW}  Skipping Azure connectivity checks — fix env vars first.{RESET}")
 
    sys.exit(print_summary())
 
 
if __name__ == "__main__":
    main()