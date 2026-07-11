from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    HealthResponse,
    DocumentInfo,
)
from app.ingestion import ingest_document
from app.retrieval import retrieve
from app.generation import generate_answer
from app.vectorstore import vector_store

app = FastAPI(
    title="DocuMind API",
    description="A RAG-based Q&A service: ask questions over your own documents.",
    version="1.0.0",
)

# Wide-open CORS for a portfolio project. Tighten to your deployed
# frontend's origin before treating this as production-grade.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def get_session_id(x_session_id: str = Header(...)) -> str:
    """
    Every request must carry an X-Session-Id header identifying the
    caller. This is NOT authentication — it doesn't prove who someone
    is — but it does give each browser/client its own isolated set of
    documents in the shared vector store, so one user of a deployed
    instance can never see, query, or delete another user's uploads.

    The frontend generates a random UUID once and persists it in
    localStorage; every request carries it.
    """
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header.")
    return x_session_id.strip()


@app.get("/health", response_model=HealthResponse)
def health(session_id: str = Header(None, alias="X-Session-Id")) -> HealthResponse:
    # Health check works with or without a session so uptime monitors
    # and the frontend's initial connectivity probe don't need one.
    count = vector_store.count(session_id) if session_id else vector_store.count()
    return HealthResponse(status="ok", documents_indexed=count)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...), session_id: str = Header(..., alias="X-Session-Id")
) -> IngestResponse:
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    chunk_count = ingest_document(file.filename, file_bytes, session_id=session_id)
    if chunk_count == 0:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this file.",
        )

    return IngestResponse(
        source=file.filename,
        chunks_created=chunk_count,
        message=f"Indexed '{file.filename}' into {chunk_count} chunks.",
    )


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(session_id: str = Header(..., alias="X-Session-Id")) -> list[DocumentInfo]:
    counts = vector_store.list_sources(session_id=session_id)
    return [DocumentInfo(source=src, chunk_count=n) for src, n in counts.items()]


@app.delete("/documents/{source}")
def delete_document(source: str, session_id: str = Header(..., alias="X-Session-Id")) -> dict:
    vector_store.delete_source(source, session_id=session_id)
    return {"message": f"Removed all chunks for '{source}'."}


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest, session_id: str = Header(..., alias="X-Session-Id")
) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(request.question, session_id=session_id, top_k=request.top_k)
    answer = generate_answer(request.question, chunks)

    return QueryResponse(
        answer=answer,
        sources=[
            SourceChunk(text=c["text"], source=c["source"], score=c["score"])
            for c in chunks
        ],
    )
