from typing import List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    source: str
    chunks_created: int
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceChunk(BaseModel):
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
