"""
core/document_processor.py
───────────────────────────
Orchestrates the full document upload pipeline:
 
  PDF bytes
    → extract text (pypdf)
    → detect section headings
    → chunk text (utils/chunker)
    → classify document type (openai_service)
    → generate embeddings per chunk (openai_service)
    → build index-ready dicts (exact Azure index field names)
 
This module is the single entry point for Tab 1 (upload flow).
It returns structured chunk dicts ready for search_service.index_chunks().
 
Index field names used here (must match Azure index exactly):
    id, content, section, type, source, contentVector,
"""
 
from __future__ import annotations
 
import logging
import re
from dataclasses import dataclass
from io import BytesIO
 
import pypdf
 
from services.openai_service import (
    classify_document,
    generate_embedding,
)
from utils.chunker import chunk_text, TextChunk
from utils.helpers import generate_chunk_id
from utils.prompt_loader import load_prompt
 
logger = logging.getLogger(__name__)
 
 
# ── Result Model ──────────────────────────────────────────────
 
@dataclass
class ProcessedDocument:
    """
    Result object returned after a document is fully processed.
    Contains everything needed for the UI to display a summary.
    """
    source: str             # Original filename
    doc_type: str           # "sds" or "tds"
    total_chunks: int       # Number of chunks produced
    total_pages: int        # Pages in the PDF
    chunks: list[dict]      # Index-ready chunk dicts
 
 
# ── PDF Text Extraction ───────────────────────────────────────
 
def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """
    Extract all text from a PDF file using pypdf.
 
    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.
 
    Returns:
        Tuple of (full_text: str, page_count: int).
 
    Raises:
        ValueError: If the PDF has no extractable text (e.g. scanned image PDF).
    """
    reader = pypdf.PdfReader(BytesIO(pdf_bytes))
    page_count = len(reader.pages)
 
    pages_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text.strip())
 
    full_text = "\n\n".join(p for p in pages_text if p)
 
    if not full_text.strip():
        raise ValueError(
            "No extractable text found in this PDF. "
            "Scanned image PDFs are not currently supported."
        )
 
    logger.info(f"Extracted {len(full_text)} characters from {page_count} pages.")
    return full_text, page_count
 
 
# ── Section Detection ─────────────────────────────────────────
 
# Common SDS section heading patterns (GHS standard 1-16 + TDS patterns)
_SECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(SECTION\s+\d+[\.\:]?\s*.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(\d+[\.\)]\s+[A-Z][A-Z\s]{3,})$", re.MULTILINE),
    re.compile(r"^([A-Z][A-Z\s\/]{5,40})$", re.MULTILINE),
]
 
 
def _detect_section(text_chunk: str, full_text: str) -> str:
    """
    Attempt to identify the section heading that precedes this chunk
    by scanning the lines just before the chunk content in the full text.
 
    Args:
        text_chunk: The chunk content string.
        full_text:  The complete document text.
 
    Returns:
        Section label string, or "General" if no heading detected.
    """
    # Find where this chunk starts in the full text
    chunk_start = full_text.find(text_chunk[:80])
    if chunk_start == -1:
        return "General"
 
    # Look at the 400 characters before this chunk for a heading
    preceding = full_text[max(0, chunk_start - 400): chunk_start]
 
    last_heading = "General"
    for pattern in _SECTION_PATTERNS:
        for match in pattern.finditer(preceding):
            last_heading = match.group(1).strip()
 
    return last_heading[:120]  # Cap section label length
 
 
# ── Full Processing Pipeline ──────────────────────────────────
 
def process_document(
    pdf_bytes: bytes,
    filename: str,
    user_id: str,
) -> ProcessedDocument:
    """
    Run the complete document processing pipeline.
 
    Steps:
      1. Extract text from PDF
      2. Classify document as SDS or TDS using LLM
      3. Chunk the text
      4. Detect section label for each chunk
      5. Generate embedding for each chunk
      6. Build index-ready chunk dicts
 
    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.
        filename:  Original filename (used as 'source' in index).
 
    Returns:
        ProcessedDocument containing all index-ready chunk dicts.
    """
    # ── Step 1: Extract text ──────────────────────────────────
    logger.info(f"[1/5] Extracting text from '{filename}'...")
    full_text, page_count = extract_text_from_pdf(pdf_bytes)
 
    # ── Step 2: Classify document type ───────────────────────
    logger.info(f"[2/5] Classifying document type...")
    classifier_prompt = load_prompt("document_classifier.md")
    doc_type = classify_document(
        document_text=full_text,
        classifier_prompt=classifier_prompt,
    )
    logger.info(f"Document classified as: '{doc_type}'")
 
    # ── Step 3: Chunk text ────────────────────────────────────
    logger.info(f"[3/5] Chunking text...")
    text_chunks: list[TextChunk] = chunk_text(full_text)
    logger.info(f"Produced {len(text_chunks)} chunks.")
 
    if not text_chunks:
        raise ValueError(f"No chunks produced from '{filename}'. Document may be too short.")
 
    # ── Step 4 & 5: Detect sections + embed + build dicts ────
    logger.info(f"[4/5] Detecting sections and generating embeddings...")
    index_docs: list[dict] = []
 
    for chunk in text_chunks:
        section = _detect_section(chunk.content, full_text)
 
        # Generate embedding for this chunk's content
        embedding = generate_embedding(chunk.content)
 
        # Build the index document — field names must match Azure index exactly
        index_doc = {
            "id":               generate_chunk_id(filename, chunk.index),
            "content":          chunk.content,
            "section":          section,
            "type":             doc_type,           # "sds" or "tds"
            "source":           filename,            # original filename
            "contentVector":    embedding,           # vector field name in Azure
        }
        index_docs.append(index_doc)
 
        logger.debug(
            f"  Chunk {chunk.index + 1}/{len(text_chunks)} | "
            f"section='{section[:40]}' | tokens={chunk.token_count}"
        )
 
    logger.info(f"[5/5] Processing complete. {len(index_docs)} chunks ready to index.")
 
    return ProcessedDocument(
        source=filename,
        doc_type=doc_type,
        total_chunks=len(index_docs),
        total_pages=page_count,
        chunks=index_docs,
    )
 
