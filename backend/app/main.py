from fastapi import FastAPI, UploadFile, File, HTTPException
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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", documents_indexed=vector_store.count())


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    chunk_count = ingest_document(file.filename, file_bytes)
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
def list_documents() -> list[DocumentInfo]:
    counts = vector_store.list_sources()
    return [DocumentInfo(source=src, chunk_count=n) for src, n in counts.items()]


@app.delete("/documents/{source}")
def delete_document(source: str) -> dict:
    vector_store.delete_source(source)
    return {"message": f"Removed all chunks for '{source}'."}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(request.question, top_k=request.top_k)
    answer = generate_answer(request.question, chunks)

    return QueryResponse(
        answer=answer,
        sources=[
            SourceChunk(text=c["text"], source=c["source"], score=c["score"])
            for c in chunks
        ],
    )
