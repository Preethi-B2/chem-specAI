"""
utils/chunker.py
─────────────────
Text chunking utilities. Pure Python — no Azure dependencies.
Uses tiktoken for token-accurate splitting to respect LLM context limits.
"""
 
from __future__ import annotations
 
import re
from dataclasses import dataclass
 
import tiktoken
 
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
 
 
# ── Data Model ────────────────────────────────────────────────
 
@dataclass
class TextChunk:
    """Represents a single chunk of text extracted from a document."""
    index: int          # Zero-based position in the document
    content: str        # The chunk text
    token_count: int    # Approximate token count
 
 
# ── Tokenizer Setup ───────────────────────────────────────────
 
def _get_tokenizer() -> tiktoken.Encoding:
    """
    Return the cl100k_base tokenizer, which is compatible with
    GPT-4, GPT-4o, and text-embedding-3-* models.
    """
    return tiktoken.get_encoding("cl100k_base")
 
 
def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string.
 
    Args:
        text: Input text string.
 
    Returns:
        Integer token count.
    """
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text))
 
 
# ── Chunking Logic ────────────────────────────────────────────
 
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split a long text string into overlapping token-based chunks.
 
    Strategy:
      1. Split text into sentences (preserve natural boundaries).
      2. Accumulate sentences into chunks up to `chunk_size` tokens.
      3. Apply `overlap` token window between consecutive chunks
         to preserve cross-chunk context.
 
    Args:
        text:       Full extracted document text.
        chunk_size: Max tokens per chunk (default from settings).
        overlap:    Token overlap between consecutive chunks (default from settings).
 
    Returns:
        List of TextChunk objects ordered by position in document.
    """
    if not text or not text.strip():
        return []
 
    tokenizer = _get_tokenizer()
    sentences = _split_into_sentences(text)
 
    chunks: list[TextChunk] = []
    current_tokens: list[int] = []
    chunk_index = 0
 
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
 
        # If adding this sentence would exceed chunk_size, flush current chunk
        if current_tokens and len(current_tokens) + len(sentence_tokens) > chunk_size:
            chunk_text_str = tokenizer.decode(current_tokens)
            chunks.append(
                TextChunk(
                    index=chunk_index,
                    content=chunk_text_str.strip(),
                    token_count=len(current_tokens),
                )
            )
            chunk_index += 1
 
            # Carry over the overlap window into the next chunk
            if overlap > 0 and len(current_tokens) > overlap:
                current_tokens = current_tokens[-overlap:]
            else:
                current_tokens = []
 
        current_tokens.extend(sentence_tokens)
 
    # Flush any remaining tokens as the final chunk
    if current_tokens:
        chunk_text_str = tokenizer.decode(current_tokens)
        chunks.append(
            TextChunk(
                index=chunk_index,
                content=chunk_text_str.strip(),
                token_count=len(current_tokens),
            )
        )
 
    return [c for c in chunks if c.content]
 
 
def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation-aware regex.
    Preserves whitespace as part of each sentence for accurate token encoding.
 
    Args:
        text: Full document text.
 
    Returns:
        List of sentence strings.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    raw_sentences = sentence_endings.split(text)
 
    # Also split on double newlines (paragraph breaks common in PDFs)
    sentences: list[str] = []
    for s in raw_sentences:
        paragraphs = re.split(r"\n{2,}", s)
        sentences.extend(p.strip() for p in paragraphs if p.strip())
 
    return sentences