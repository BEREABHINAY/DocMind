"""
Text extraction and chunking logic, kept free of any ML dependency
(no sentence-transformers, no vector DB) so it can be unit tested in
isolation and reasoned about on its own.
"""
import io
from typing import List

from pypdf import PdfReader

from app.config import settings


def extract_text(filename: str, file_bytes: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Word-based sliding-window chunking with overlap.

    Overlap exists so a sentence or fact sitting right on a chunk
    boundary still appears whole in at least one chunk, instead of
    being split across two chunks and losing retrievability in either.
    """
    chunk_size = chunk_size or settings.chunk_size_words
    overlap = overlap or settings.chunk_overlap_words
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks
