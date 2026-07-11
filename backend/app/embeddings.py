"""
Wraps the local sentence-transformers embedding model.

Loaded once, at process start — encoding is CPU-bound and reasonably
fast for the model size we use here (~80MB, 384-dim output).
"""
from typing import List
from sentence_transformers import SentenceTransformer

from app.config import settings

_model = SentenceTransformer(settings.embedding_model)


def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
