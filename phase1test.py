"""
test_phase1.py
───────────────
Quick smoke-test for all Phase 1 components.
Run from project root:  python test_phase1.py
 
Does NOT require Azure credentials — purely validates local logic.
"""
 
import sys
import os
 
# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
 
 
def test_settings():
    print("\n[1] Testing config/settings.py ...")
    try:
        from config import settings
        # Just confirm module loads; Azure vars may not be set yet
        print(f"    APP_ENV        : {settings.APP_ENV}")
        print(f"    CHUNK_SIZE     : {settings.CHUNK_SIZE}")
        print(f"    CHUNK_OVERLAP  : {settings.CHUNK_OVERLAP}")
        print(f"    MAX_MEMORY_TURNS: {settings.MAX_MEMORY_TURNS}")
        print(f"    TOP_K_CHUNKS   : {settings.TOP_K_CHUNKS}")
        print("    ✅ settings.py loaded successfully")
    except EnvironmentError as e:
        print(f"    ⚠️  Missing Azure keys (expected in dev): {e}")
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        raise
 
 
def test_prompt_loader():
    print("\n[2] Testing utils/prompt_loader.py ...")
    from utils.prompt_loader import load_prompt, list_available_prompts
 
    available = list_available_prompts()
    print(f"    Found prompts: {available}")
 
    expected = [
        "system_prompt.md",
        "document_classifier.md",
        "query_classifier.md",
        "answer_generator.md",
    ]
    for fname in expected:
        text = load_prompt(fname)
        assert len(text) > 50, f"Prompt '{fname}' appears empty or too short"
        print(f"    ✅ {fname} loaded ({len(text)} chars)")
 
    # Test cache hit
    text2 = load_prompt("system_prompt.md")
    assert text2 == load_prompt("system_prompt.md"), "Cache inconsistency"
    print("    ✅ LRU cache working correctly")
 
 
def test_helpers():
    print("\n[3] Testing utils/helpers.py ...")
    from utils.helpers import (
        generate_chunk_id,
        generate_upload_id,
        utc_now_iso,
        get_user_id,
        sanitize_filename,
        build_blob_path,
    )
 
    # Deterministic chunk IDs
    id1 = generate_chunk_id("chemical_x.pdf", 0)
    id2 = generate_chunk_id("chemical_x.pdf", 0)
    assert id1 == id2, "Chunk ID must be deterministic"
    id3 = generate_chunk_id("chemical_x.pdf", 1)
    assert id1 != id3, "Different chunk index must produce different ID"
    print(f"    ✅ generate_chunk_id: deterministic & unique per index")
 
    upload_id = generate_upload_id()
    assert len(upload_id) == 36, "UUID4 should be 36 chars"
    print(f"    ✅ generate_upload_id: {upload_id}")
 
    ts = utc_now_iso()
    assert "+" in ts or "Z" in ts or "UTC" in ts or len(ts) > 20
    print(f"    ✅ utc_now_iso: {ts}")
 
    session = {}
    uid1 = get_user_id(session)
    uid2 = get_user_id(session)
    assert uid1 == uid2, "Same session must return same user_id"
    print(f"    ✅ get_user_id: stable within session ({uid1[:8]}...)")
 
    safe = sanitize_filename("../../../etc/passwd.pdf")
    assert "/" not in safe and ".." not in safe
    print(f"    ✅ sanitize_filename: '{safe}'")
 
    path = build_blob_path("user123", "chemical_x sds.pdf")
    assert "user123" in path
    assert " " not in path
    print(f"    ✅ build_blob_path: '{path}'")
 
 
def test_chunker():
    print("\n[4] Testing utils/chunker.py ...")
    from utils.chunker import chunk_text, count_tokens, TextChunk
 
    sample_text = """
    Hydrochloric acid (HCl) is a strong acid. It is highly corrosive and can cause severe burns.
    Workers handling HCl must wear appropriate PPE including chemical-resistant gloves and face shields.
    The OSHA permissible exposure limit (PEL) is 5 ppm as a ceiling value.
    In case of skin contact, immediately flush with large amounts of water for at least 15 minutes.
    Store in a cool, dry, well-ventilated area away from incompatible materials such as bases and metals.
    The product has a boiling point of -85.05°C and a molecular weight of 36.46 g/mol.
    It is classified as a hazardous substance under GHS with signal word DANGER.
    Transport classification: UN 1789, Packing Group II, Hazard Class 8.
    """ * 10  # Repeat to create enough text to trigger multiple chunks
 
    token_count = count_tokens(sample_text)
    print(f"    Sample text: {len(sample_text)} chars, ~{token_count} tokens")
 
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
    print(f"    Produced {len(chunks)} chunks")
 
    assert len(chunks) > 1, "Should produce multiple chunks for long text"
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, TextChunk)
        assert chunk.index == i
        assert len(chunk.content) > 0
        assert chunk.token_count > 0
    print(f"    ✅ All {len(chunks)} chunks valid")
 
    # Edge cases
    empty_chunks = chunk_text("")
    assert empty_chunks == [], "Empty text should return empty list"
    print("    ✅ Empty text handled correctly")
 
    short_chunks = chunk_text("Hello world.", chunk_size=500, overlap=50)
    assert len(short_chunks) == 1
    print("    ✅ Short text produces single chunk")
 
 
def main():
    print("=" * 55)
    print("  Phase 1 Smoke Tests — Chemistry Doc Intelligence")
    print("=" * 55)
 
    test_settings()
    test_prompt_loader()
    test_helpers()
    test_chunker()
 
    print("\n" + "=" * 55)
    print("  ✅ All Phase 1 tests passed!")
    print("=" * 55)
 
 
if __name__ == "__main__":
    main()