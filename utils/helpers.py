"""
utils/helpers.py
─────────────────
Stateless utility functions: ID generation, timestamps, user identity.
No Azure dependencies — safe to import and test anywhere.
"""
 
from __future__ import annotations
import os
import hashlib
import uuid
from datetime import datetime, timezone
 
 
# ── ID & Timestamp Generators ────────────────────────────────
 
def generate_chunk_id(file_name: str, chunk_index: int) -> str:
    """
    Generate a deterministic, unique chunk ID based on file name and index.
    Deterministic means re-indexing the same file produces the same IDs,
    which allows safe upserts into Azure AI Search.
 
    Args:
        file_name:    Original filename, e.g. "chemical_x_sds.pdf"
        chunk_index:  Zero-based position of the chunk within the document.
 
    Returns:
        Hex digest string, e.g. "a3f9e2..."
    """
    #raw = f"{file_name}::{chunk_index}"
    base_name = os.path.splitext(file_name)[0]
    raw = f"{base_name}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()
 
 
def generate_upload_id() -> str:
    """
    Generate a random UUID for each upload session.
 
    Returns:
        UUID4 string, e.g. "550e8400-e29b-41d4-a716-446655440000"
    """
    return str(uuid.uuid4())
 
 
def utc_now_iso() -> str:
    """
    Return the current UTC timestamp in ISO 8601 format.
 
    Returns:
        e.g. "2026-03-04T10:30:00.123456+00:00"
    """
    return datetime.now(tz=timezone.utc).isoformat()
 
 
# ── User Identity ─────────────────────────────────────────────
 
def get_user_id(session_state: dict) -> str:
    """
    Retrieve or create a stable user_id from Streamlit session state.
    In a real enterprise system this would come from Azure AD / SSO.
    For now, a UUID is generated once per browser session and reused.
 
    Args:
        session_state: Streamlit's st.session_state dict.
 
    Returns:
        Stable user_id string for this session.
    """
    if "user_id" not in session_state:
        session_state["user_id"] = str(uuid.uuid4())
    return session_state["user_id"]
 
 
# ── Sanitization ─────────────────────────────────────────────
 
def sanitize_filename(filename: str) -> str:
    """
    Strip path separators and whitespace from a filename to prevent
    directory traversal when constructing blob storage paths.
 
    Args:
        filename: Raw filename from file uploader.
 
    Returns:
        Safe filename string.
    """
    return (
        filename.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("..", "_")
    )
 
 
def build_blob_path(user_id: str, filename: str) -> str:
    """
    Construct the Azure Blob Storage path for a user's uploaded document.
    Organizes files per user to enforce isolation.
 
    Args:
        user_id:  Stable session user identifier.
        filename: Sanitized filename.
 
    Returns:
        Blob path string, e.g. "user_abc123/2026-03-04T10:30:00_chemical_x.pdf"
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_name = sanitize_filename(filename)
    return f"{user_id}/{timestamp}_{safe_name}"