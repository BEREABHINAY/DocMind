from typing import List, Dict, Any

from app.config import settings
from app.embeddings import embed_query
from app.vectorstore import vector_store


def retrieve(question: str, session_id: str, top_k: int = None) -> List[Dict[str, Any]]:
    top_k = top_k or settings.top_k
    query_vector = embed_query(question)
    return vector_store.search(query_vector, top_k=top_k, session_id=session_id)
