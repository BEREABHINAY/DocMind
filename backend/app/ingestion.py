"""
Turns a raw file into searchable chunks.

Pipeline: extract text -> split into overlapping word-windows ->
embed each chunk -> store in the vector database. This module owns
the "index this document" half of RAG; retrieval.py owns the other half.
"""
import time

from app.chunking import extract_text, chunk_text
from app.embeddings import embed_texts
from app.vectorstore import vector_store


def ingest_document(filename: str, file_bytes: bytes) -> int:
    text = extract_text(filename, file_bytes)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    base_id = int(time.time() * 1000)
    ids = [base_id + i for i in range(len(chunks))]
    payloads = [{"text": c, "source": filename} for c in chunks]

    vector_store.upsert_chunks(ids=ids, vectors=vectors, payloads=payloads)
    return len(chunks)
