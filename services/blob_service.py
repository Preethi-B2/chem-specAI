"""
services/blob_service.py
─────────────────────────
Azure Blob Storage wrapper for:
  - Uploading raw PDF documents
  - Downloading documents (for re-processing)
  - Listing documents per user
  - Deleting documents
 
All blobs are organized as:  {user_id}/{timestamp}_{filename}
This enforces per-user folder isolation at the storage level.
"""
 
from __future__ import annotations
 
import logging
from datetime import datetime, timezone
from io import BytesIO
 
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings
 
from config.settings import (
    AZURE_BLOB_CONNECTION_STRING,
    AZURE_BLOB_CONTAINER_NAME,
)
 
logger = logging.getLogger(__name__)
 
 
# ── Client (singleton per module) ────────────────────────────
 
def _get_client() -> BlobServiceClient:
    """Return a configured BlobServiceClient."""
    if not AZURE_BLOB_CONNECTION_STRING:
        raise EnvironmentError(
            "AZURE_BLOB_CONNECTION_STRING is not set in your .env file. "
            "Get it from Azure Portal → Storage Accounts → Access keys → Connection string."
        )
    return BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
 
 
def _get_container_client():
    """Return a ContainerClient for the chemistry-docs container."""
    return _get_client().get_container_client(AZURE_BLOB_CONTAINER_NAME)
 
 
# ── Container Bootstrap ───────────────────────────────────────
 
def ensure_container_exists() -> None:
    """
    Create the blob container if it does not already exist.
    Safe to call on every app startup.
    """
    try:
        container_client = _get_container_client()
        container_client.create_container()
        logger.info(f"Container '{AZURE_BLOB_CONTAINER_NAME}' created.")
    except Exception as e:
        # ResourceExistsError is expected on subsequent startups — ignore it
        if "ContainerAlreadyExists" in str(e) or "already exists" in str(e).lower():
            logger.debug(f"Container '{AZURE_BLOB_CONTAINER_NAME}' already exists.")
        else:
            logger.error(f"Failed to ensure container exists: {e}")
            raise
 
 
# ── Upload ────────────────────────────────────────────────────
 
def upload_document(
    file_bytes: bytes,
    blob_path: str,
    content_type: str = "application/pdf",
) -> str:
    """
    Upload a raw document (PDF) to Azure Blob Storage.
 
    Args:
        file_bytes:   Raw bytes of the uploaded file.
        blob_path:    Full blob path including user folder,
                      e.g. "user_abc/20260304T103000_chemical_x_sds.pdf"
        content_type: MIME type of the file.
 
    Returns:
        The full blob URL as a string.
 
    Raises:
        AzureError: On upload failure.
    """
    try:
        container_client = _get_container_client()
        blob_client = container_client.get_blob_client(blob_path)
 
        blob_client.upload_blob(
            data=BytesIO(file_bytes),
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
 
        url = blob_client.url
        logger.info(f"Uploaded blob: {blob_path} ({len(file_bytes)} bytes)")
        return url
 
    except AzureError as e:
        logger.error(f"Blob upload failed for '{blob_path}': {e}")
        raise
 
 
# ── Download ──────────────────────────────────────────────────
 
def download_document(blob_path: str) -> bytes:
    """
    Download a document from Azure Blob Storage.
 
    Args:
        blob_path: Full blob path, e.g. "user_abc/20260304T103000_chemical_x.pdf"
 
    Returns:
        Raw bytes of the file.
 
    Raises:
        ResourceNotFoundError: If the blob does not exist.
        AzureError: On download failure.
    """
    try:
        container_client = _get_container_client()
        blob_client = container_client.get_blob_client(blob_path)
        download = blob_client.download_blob()
        data = download.readall()
        logger.info(f"Downloaded blob: {blob_path} ({len(data)} bytes)")
        return data
 
    except ResourceNotFoundError:
        logger.error(f"Blob not found: '{blob_path}'")
        raise
    except AzureError as e:
        logger.error(f"Blob download failed for '{blob_path}': {e}")
        raise
 
 
# ── List ──────────────────────────────────────────────────────
 
def list_user_documents(user_id: str) -> list[dict]:
    """
    List all blobs uploaded by a specific user.
 
    Args:
        user_id: The user's session identifier (used as blob folder prefix).
 
    Returns:
        List of dicts with keys: name, size, last_modified, url
    """
    try:
        container_client = _get_container_client()
        prefix = f"{user_id}/"
        blobs = container_client.list_blobs(name_starts_with=prefix)
 
        results = []
        for blob in blobs:
            results.append({
                "name": blob.name,
                "size": blob.size,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                "url": f"{container_client.url}/{blob.name}",
            })
 
        logger.info(f"Listed {len(results)} blobs for user '{user_id}'")
        return results
 
    except AzureError as e:
        logger.error(f"Failed to list blobs for user '{user_id}': {e}")
        raise
 
 
# ── Delete ────────────────────────────────────────────────────
 
def delete_document(blob_path: str) -> None:
    """
    Delete a blob from Azure Blob Storage.
 
    Args:
        blob_path: Full blob path to delete.
 
    Raises:
        ResourceNotFoundError: If the blob does not exist.
        AzureError: On deletion failure.
    """
    try:
        container_client = _get_container_client()
        blob_client = container_client.get_blob_client(blob_path)
        blob_client.delete_blob()
        logger.info(f"Deleted blob: {blob_path}")
 
    except ResourceNotFoundError:
        logger.warning(f"Blob not found for deletion: '{blob_path}'")
        raise
    except AzureError as e:
        logger.error(f"Blob deletion failed for '{blob_path}': {e}")
        raise
 