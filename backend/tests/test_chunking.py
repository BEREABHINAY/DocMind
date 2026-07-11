import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_text  # noqa: E402


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_single_chunk():
    text = "This is a short sentence with few words."
    chunks = chunk_text(text, chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_splits_into_multiple_chunks():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # every original word must still appear in the reconstructed chunks
    all_chunk_words = " ".join(chunks).split()
    assert set(words).issubset(set(all_chunk_words))


def test_overlap_preserves_boundary_content():
    """A phrase sitting right on a chunk boundary should survive whole
    in at least one chunk, which is the entire point of overlapping."""
    words = [f"w{i}" for i in range(100)]
    boundary_phrase = "IMPORTANT_FACT_HERE"
    words[98] = boundary_phrase
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=60, overlap=15)
    assert any(boundary_phrase in c for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=10, overlap=10)


def test_no_word_is_ever_dropped_entirely():
    words = [f"tok{i}" for i in range(237)]  # odd, non-round number on purpose
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    covered = set(" ".join(chunks).split())
    assert covered == set(words)
