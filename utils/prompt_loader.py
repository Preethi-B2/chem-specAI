"""
utils/prompt_loader.py
───────────────────────
Dynamically loads prompt text from .md files under the /prompts directory.
No prompts are ever hardcoded in application logic — all live in .md files.
"""
 
from __future__ import annotations
 
import os
from functools import lru_cache
from pathlib import Path
 
# Resolve the prompts directory relative to project root
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
 
 
@lru_cache(maxsize=None)
def load_prompt(filename: str) -> str:
    """
    Load and return the contents of a prompt .md file.
 
    Args:
        filename: Name of the .md file, e.g. "system_prompt.md"
 
    Returns:
        Full text content of the prompt file as a string.
 
    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    prompt_path = _PROMPTS_DIR / filename
 
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"[prompt_loader] Prompt file not found: '{prompt_path}'. "
            f"Available prompts: {list_available_prompts()}"
        )
 
    return prompt_path.read_text(encoding="utf-8").strip()
 
 
def reload_prompt(filename: str) -> str:
    """
    Force-reload a prompt file, bypassing the LRU cache.
    Useful during development when prompt files are actively being edited.
 
    Args:
        filename: Name of the .md file to reload.
 
    Returns:
        Fresh text content from disk.
    """
    load_prompt.cache_clear()
    return load_prompt(filename)
 
 
def list_available_prompts() -> list[str]:
    """
    Return a list of all .md files in the prompts directory.
 
    Returns:
        List of filenames (not full paths), e.g. ["system_prompt.md", ...]
    """
    if not _PROMPTS_DIR.exists():
        return []
    return sorted(p.name for p in _PROMPTS_DIR.glob("*.md"))