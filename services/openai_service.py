"""
services/openai_service.py
───────────────────────────
Azure OpenAI wrapper for:
  - Chat completions (LLM calls)
  - Embedding generation
  - Document classification
  - Query classification
 
Uses the official openai>=1.30.0 SDK with AzureOpenAI client.
All prompts are injected externally — no hardcoded strings here.
Retry logic via tenacity for transient Azure errors.
"""
 
from __future__ import annotations
 
import logging
from typing import Any
 
from openai import AzureOpenAI, APIStatusError, APIConnectionError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
 
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DIMENSIONS,
)
 
logger = logging.getLogger(__name__)
 
# ── Retryable Azure error types ──────────────────────────────
_RETRYABLE = (APIConnectionError, APITimeoutError)
 
 
# ── Client (singleton per module) ────────────────────────────
def _get_client() -> AzureOpenAI:
    """
    Return a configured AzureOpenAI client.
    Created once and reused across calls within the same process.
    """
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
 
 
# ── Embedding Generation ──────────────────────────────────────
 
@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def generate_embedding(text: str) -> list[float]:
    """
    Generate a vector embedding for a single text string.
 
    Args:
        text: Input text to embed (typically a document chunk).
 
    Returns:
        List of floats representing the embedding vector.
 
    Raises:
        APIStatusError: On non-retryable Azure API errors.
    """
    client = _get_client()
 
    # Azure text-embedding-3-large supports dimensions parameter
    response = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text.replace("\n", " "),  # newlines degrade embedding quality
        dimensions=AZURE_OPENAI_EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding
 
 
def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    Processes individually to handle per-item errors gracefully.
 
    Args:
        texts: List of text strings (e.g., document chunks).
 
    Returns:
        List of embedding vectors in the same order as input.
    """
    embeddings: list[list[float]] = []
    for i, text in enumerate(texts):
        logger.debug(f"Embedding chunk {i + 1}/{len(texts)}")
        embeddings.append(generate_embedding(text))
    return embeddings
 
 
# ── Chat Completion ───────────────────────────────────────────
 
@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def chat_completion(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict = "auto",
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> Any:
    """
    Call Azure OpenAI chat completion endpoint.
 
    Args:
        messages:     Full conversation messages list in OpenAI format.
        tools:        Optional list of tool definitions for tool calling.
        tool_choice:  "auto" | "none" | {"type": "function", "function": {"name": "..."}}
        temperature:  Sampling temperature. 0.0 = deterministic (recommended for RAG).
        max_tokens:   Maximum tokens in the response.
 
    Returns:
        The raw OpenAI ChatCompletion response object.
        Callers extract .choices[0].message as needed.
    """
    client = _get_client()
 
    kwargs: dict[str, Any] = {
        "model": AZURE_OPENAI_CHAT_DEPLOYMENT,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
 
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
 
    return client.chat.completions.create(**kwargs)
 
 
# ── Document Classification ───────────────────────────────────
 
def classify_document(document_text: str, classifier_prompt: str) -> str:
    """
    Use the LLM to classify a document as 'sds' or 'tds'.
 
    Args:
        document_text:      Extracted text from the uploaded PDF.
                            Truncated to first 3000 chars to stay within limits.
        classifier_prompt:  Loaded content of document_classifier.md
                            with {document_text} placeholder.
 
    Returns:
        "sds" or "tds" (lowercased, stripped).
 
    Raises:
        ValueError: If the LLM returns an unexpected classification.
    """
    # Truncate — we only need the beginning of the doc to classify
    truncated_text = document_text[:3000].strip()
 
    filled_prompt = classifier_prompt.replace("{document_text}", truncated_text)
 
    messages = [
        {"role": "user", "content": filled_prompt},
    ]
 
    response = chat_completion(messages=messages, temperature=0.0, max_tokens=10)
    raw = response.choices[0].message.content.strip().lower()
 
    if raw not in ("sds", "tds"):
        logger.warning(f"Unexpected classification response: '{raw}'. Defaulting to 'tds'.")
        return "tds"
 
    logger.info(f"Document classified as: {raw}")
    return raw
 
 
# ── Query Classification ──────────────────────────────────────
 
def classify_query(
    user_question: str,
    conversation_history: str,
    query_classifier_prompt: str,
) -> str:
    """
    Use the LLM to classify a user query as 'sds' or 'tds'
    for intelligent retrieval routing.
 
    Args:
        user_question:           The current user query.
        conversation_history:    Formatted string of prior turns for context.
        query_classifier_prompt: Loaded content of query_classifier.md.
 
    Returns:
        "sds" or "tds" (lowercased, stripped).
    """
    filled_prompt = (
        query_classifier_prompt
        .replace("{user_question}", user_question)
        .replace("{conversation_history}", conversation_history or "No prior conversation.")
    )
 
    messages = [{"role": "user", "content": filled_prompt}]
 
    response = chat_completion(messages=messages, temperature=0.0, max_tokens=10)
    raw = response.choices[0].message.content.strip().lower()
 
    if raw not in ("sds", "tds"):
        logger.warning(f"Unexpected query classification: '{raw}'. Defaulting to 'sds'.")
        return "sds"
 
    logger.info(f"Query classified as: {raw}")
    return raw