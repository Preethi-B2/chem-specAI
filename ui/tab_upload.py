"""
ui/tab_upload.py
─────────────────
Tab 1 — Document Upload
 
Responsibilities (UI only — no Azure calls directly):
  1. Render file uploader widget
  2. Validate file type and size
  3. Call core/document_processor.py → process_document()
  4. Call services/blob_service.py   → upload_document()
  5. Call services/search_service.py → index_chunks()
  6. Display processing progress and result summary
 
Rule: This module only handles rendering and user interaction.
All business logic lives in core/ and services/.
"""
 
from __future__ import annotations
 
import logging
 
import streamlit as st
 
from core.document_processor import process_document, ProcessedDocument
from services.blob_service import upload_document, ensure_container_exists
from services.search_service import index_chunks, ensure_index_exists
from utils.helpers import get_user_id, build_blob_path, sanitize_filename
 
logger = logging.getLogger(__name__)
 
# Max upload size: 20 MB
_MAX_FILE_SIZE_MB = 20
_MAX_FILE_SIZE_BYTES = _MAX_FILE_SIZE_MB * 1024 * 1024
 
 
def render_upload_tab() -> None:
    """
    Render the full Tab 1 — Document Upload interface.
    Called directly from app.py inside the st.tabs() block.
    """
    user_id = get_user_id(st.session_state)
 
    st.header("📂 Document Upload")
    st.caption(
        "Upload a chemistry PDF (SDS or TDS). "
        "The system will automatically classify, chunk, embed, and index it."
    )
 
    # ── Bootstrap Azure resources ─────────────────────────────
    _bootstrap_azure()
 
    # ── File uploader ─────────────────────────────────────────
    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],
        help=f"Maximum file size: {_MAX_FILE_SIZE_MB} MB",
        key="upload_file_input",
    )
 
    if uploaded_file is None:
        _render_empty_state()
        return
 
    # ── Validate file ─────────────────────────────────────────
    file_bytes = uploaded_file.read()
 
    if len(file_bytes) == 0:
        st.error("⚠️ The uploaded file is empty. Please upload a valid PDF.")
        return
 
    if len(file_bytes) > _MAX_FILE_SIZE_BYTES:
        st.error(
            f"⚠️ File size ({len(file_bytes) / 1024 / 1024:.1f} MB) "
            f"exceeds the {_MAX_FILE_SIZE_MB} MB limit."
        )
        return
 
    safe_filename = sanitize_filename(uploaded_file.name)
 
    # ── File preview card ─────────────────────────────────────
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("File Name",  safe_filename)
        col2.metric("Size",       f"{len(file_bytes) / 1024:.1f} KB")
        col3.metric("Type",       "PDF")
 
    st.divider()
 
    # ── Process button ────────────────────────────────────────
    if st.button("🚀 Process & Index Document", type="primary", use_container_width=True):
        _run_upload_pipeline(
            file_bytes=file_bytes,
            filename=safe_filename,
            user_id=user_id,
        )
 
 
def _run_upload_pipeline(
    file_bytes: bytes,
    filename: str,
    user_id: str,
) -> None:
    """
    Execute the full upload → process → index pipeline with live progress UI.
 
    Args:
        file_bytes: Raw PDF bytes.
        filename:   Sanitized filename.
        user_id:    Current session user ID.
    """
    progress_bar = st.progress(0, text="Starting...")
    status_area  = st.empty()
 
    try:
        # ── Step 1: Upload to Blob Storage ────────────────────
        status_area.info("📤 **Step 1/4** — Uploading to Azure Blob Storage...")
        progress_bar.progress(10, text="Uploading to Azure Blob Storage...")
 
        blob_path = build_blob_path(user_id, filename)
        blob_url  = upload_document(
            file_bytes=file_bytes,
            blob_path=blob_path,
        )
        logger.info(f"Blob uploaded: {blob_url}")
 
        # ── Step 2: Extract + Classify + Chunk + Embed ────────
        status_area.info("🔬 **Step 2/4** — Extracting text and classifying document...")
        progress_bar.progress(30, text="Processing document...")
 
        processed: ProcessedDocument = process_document(
            pdf_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
        )
 
        # ── Step 3: Update progress after embedding ───────────
        status_area.info(
            f"🔢 **Step 3/4** — Generated embeddings for "
            f"{processed.total_chunks} chunks..."
        )
        progress_bar.progress(75, text="Embeddings generated...")
 
        # ── Step 4: Index chunks in Azure AI Search ───────────
        status_area.info("📇 **Step 4/4** — Indexing chunks in Azure AI Search...")
        progress_bar.progress(90, text="Indexing in Azure AI Search...")
 
        indexed_count = index_chunks(processed.chunks)
 
        # ── Done ──────────────────────────────────────────────
        progress_bar.progress(100, text="Complete!")
        status_area.empty()
 
        _render_success_summary(processed, indexed_count, blob_url)
 
    except ValueError as e:
        progress_bar.empty()
        status_area.empty()
        st.error(f"⚠️ Document processing error: {e}")
        logger.error(f"Upload pipeline ValueError: {e}")
 
    except Exception as e:
        progress_bar.empty()
        status_area.empty()
        st.error(
            f"❌ An unexpected error occurred: {e}\n\n"
            "Please check your Azure configuration and try again."
        )
        logger.exception(f"Upload pipeline failed for '{filename}': {e}")
 
 
def _render_success_summary(
    processed: ProcessedDocument,
    indexed_count: int,
    blob_url: str,
) -> None:
    """
    Render the success summary card after a document is fully processed.
 
    Args:
        processed:     ProcessedDocument result from document_processor.
        indexed_count: Number of chunks successfully indexed.
        blob_url:      Azure Blob Storage URL of the raw file.
    """
    doc_type_label = "🟠 Safety Data Sheet (SDS)" if processed.doc_type == "sds" \
                     else "🔵 Technical Data Sheet (TDS)"
    doc_type_color = "orange" if processed.doc_type == "sds" else "blue"
 
    st.success("✅ Document processed and indexed successfully!")
 
    with st.container(border=True):
        st.subheader("📋 Processing Summary")
 
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Document Type",   doc_type_label)
        col2.metric("Pages Parsed",    processed.total_pages)
        col3.metric("Chunks Created",  processed.total_chunks)
        col4.metric("Chunks Indexed",  indexed_count)
 
        st.divider()
 
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**📄 File:** `{processed.source}`")
            st.markdown(f"**🏷️ Type:** :{doc_type_color}[{processed.doc_type.upper()}]")
        with col_b:
            st.markdown("**☁️ Stored in:** Azure Blob Storage")
            st.markdown("**🔍 Indexed in:** Azure AI Search")
 
        if indexed_count < processed.total_chunks:
            st.warning(
                f"⚠️ {processed.total_chunks - indexed_count} chunk(s) failed to index. "
                "Partial results may be available."
            )
 
    st.info(
        "💬 Switch to the **Chat** tab to start asking questions about this document.",
        icon="💡",
    )
 
 
def _render_empty_state() -> None:
    """Render the empty/idle state before any file is uploaded."""
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 3rem 1rem;
            border: 2px dashed #ccc;
            border-radius: 12px;
            color: #888;
        ">
            <h3>📄 No document uploaded yet</h3>
            <p>Upload a chemistry PDF above to get started.</p>
            <p><small>Supported: Safety Data Sheets (SDS) · Technical Data Sheets (TDS)</small></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
def _bootstrap_azure() -> None:
    """
    Ensure Azure Blob container and AI Search index exist.
    Runs once per session using session_state flag.
    """
    if st.session_state.get("azure_bootstrapped"):
        return
    try:
        with st.spinner("Connecting to Azure services..."):
            ensure_container_exists()
            ensure_index_exists()
        st.session_state["azure_bootstrapped"] = True
    except Exception as e:
        st.error(
            f"❌ Failed to connect to Azure services: {e}\n\n"
            "Check your `.env` configuration."
        )
        logger.exception("Azure bootstrap failed.")
        st.stop()