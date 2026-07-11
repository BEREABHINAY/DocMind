"""
Thin wrapper around Qdrant.

Runs in local/embedded mode out of the box (no external service needed),
but switches to a remote Qdrant instance automatically the moment
QDRANT_URL is set — e.g. when you point it at a free Qdrant Cloud
cluster for a real deployment. Same code path either way.

Every document is tagged with a session_id at ingest time, and every
read/delete operation is filtered by session_id, so one user's
uploaded documents are never visible to another user sharing the same
deployed instance.
"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings


def _session_filter(session_id: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
    )


class VectorStore:
    def __init__(self) -> None:
        if settings.qdrant_url:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
        else:
            self._client = QdrantClient(path=settings.qdrant_local_path)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if settings.collection_name not in existing:
            self._client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim, distance=Distance.COSINE
                ),
            )

    def upsert_chunks(
        self, ids: List[int], vectors: List[List[float]], payloads: List[Dict[str, Any]]
    ) -> None:
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self._client.upsert(collection_name=settings.collection_name, points=points)

    def search(
        self, query_vector: List[float], top_k: int, session_id: str
    ) -> List[Dict[str, Any]]:
        response = self._client.query_points(
            collection_name=settings.collection_name,
            query=query_vector,
            query_filter=_session_filter(session_id),
            limit=top_k,
        )
        return [
            {"text": p.payload["text"], "source": p.payload["source"], "score": p.score}
            for p in response.points
        ]

    def count(self, session_id: Optional[str] = None) -> int:
        if session_id is None:
            info = self._client.get_collection(settings.collection_name)
            return info.points_count or 0
        result = self._client.count(
            collection_name=settings.collection_name,
            count_filter=_session_filter(session_id),
        )
        return result.count

    def list_sources(self, session_id: str) -> Dict[str, int]:
        """Scroll this session's chunks and tally counts per source document."""
        counts: Dict[str, int] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=settings.collection_name,
                scroll_filter=_session_filter(session_id),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                src = p.payload.get("source", "unknown")
                counts[src] = counts.get(src, 0) + 1
            if offset is None:
                break
        return counts

    def delete_source(self, source: str, session_id: str) -> None:
        self._client.delete(
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source)),
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                ]
            ),
        )


vector_store = VectorStore()
