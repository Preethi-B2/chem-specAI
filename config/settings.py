"""
config/settings.py
──────────────────
Single source of truth for all environment-driven configuration.
All other modules import from here — never from os.environ directly.
"""
 
import os
from dotenv import load_dotenv
 
# Load .env once at import time
load_dotenv()
 
 
def _require(key: str) -> str:
    """Fetch a required env var; raise clearly if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"[settings] Required environment variable '{key}' is not set. "
            f"Check your .env file."
        )
    return value
 
 
def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)
 
 
# ── Azure OpenAI ────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT: str          = _require("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY: str           = _require("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION: str       = _optional("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_CHAT_DEPLOYMENT: str   = _optional("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = _optional(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)
AZURE_OPENAI_EMBEDDING_DIMENSIONS: int = int(
    _optional("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "3072")
)
 
# ── Azure AI Search ─────────────────────────────────────────
AZURE_SEARCH_ENDPOINT: str    = _require("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY: str     = _require("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME: str  = _optional("AZURE_SEARCH_INDEX_NAME", "chemistry-docs-index")
 
# ── Azure Blob Storage ──────────────────────────────────────
AZURE_BLOB_CONNECTION_STRING: str = _require("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME: str    = _optional("AZURE_BLOB_CONTAINER_NAME", "chemistry-docs")
 
# ── Application ─────────────────────────────────────────────
APP_ENV: str            = _optional("APP_ENV", "development")
MAX_MEMORY_TURNS: int   = int(_optional("MAX_MEMORY_TURNS", "5"))
TOP_K_CHUNKS: int       = int(_optional("TOP_K_CHUNKS", "5"))
CHUNK_SIZE: int         = int(_optional("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int      = int(_optional("CHUNK_OVERLAP", "50"))