# DocuMind — RAG-based Q&A over your own documents

Ask questions in plain English about PDFs or text files you upload, and get an
answer grounded in the actual document content — with the exact source
passages shown alongside it, not just a page reference.

This is a Retrieval-Augmented Generation (RAG) system, built end to end:
chunking → embedding → vector search → grounded generation → **evaluation**
(the part most portfolio RAG projects skip).

## Why this exists

Ctrl+F only matches exact words. If your document says *"minimum aggregate
of 60%"* and you search *"eligibility criteria"*, Ctrl+F finds nothing. This
system searches by **meaning**, not literal text, and combines information
from multiple places in a document into one direct answer.

## Architecture

```
                     ┌─────────────────────────────┐
                     │   1. INGEST (once per doc)   │
                     └─────────────────────────────┘
  PDF / .txt  ──▶  extract text  ──▶  chunk (sliding word window,
                                       with overlap so no fact gets
                                       split across a chunk boundary)
                                            │
                                            ▼
                              embed each chunk (sentence-transformers,
                                    runs locally, no API cost)
                                            │
                                            ▼
                              store in Qdrant (vector database)

                     ┌─────────────────────────────┐
                     │   2. QUERY (every question)  │
                     └─────────────────────────────┘
  question  ──▶  embed question  ──▶  vector search top-K chunks
                                            │
                                            ▼
                          stuff into prompt + call LLM (Groq / Llama 3.1)
                                            │
                                            ▼
                              answer, grounded only in retrieved text
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI (Python) | Async-ready, auto-generates OpenAPI docs at `/docs` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally, zero API cost, 384-dim |
| Vector DB | Qdrant | Runs embedded/local with zero setup; same code switches to Qdrant Cloud for production with one env var |
| LLM | Groq (Llama 3.1, via OpenAI-compatible API) | Free tier, very fast inference |
| Frontend | React + Vite | Simple chat UI with a document library sidebar |
| Evaluation | Custom retrieval hit-rate harness | Measures whether retrieval actually works, not just vibes |

## Project structure

```
documind-rag/
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI routes
│   │   ├── chunking.py     # text extraction + chunking (pure logic, no ML deps)
│   │   ├── embeddings.py   # sentence-transformers wrapper
│   │   ├── ingestion.py    # ties chunking + embedding + storage together
│   │   ├── retrieval.py    # question → embedding → vector search
│   │   ├── generation.py   # retrieved chunks + question → LLM answer
│   │   ├── vectorstore.py  # Qdrant wrapper (local or cloud)
│   │   └── config.py       # all settings, loaded from .env
│   ├── data/                # sample documents for a working demo out of the box
│   ├── tests/                # pytest suite — runs with no network needed
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── src/App.jsx          # chat UI
└── evaluation/
    ├── eval_set.json        # sample question/answer pairs
    └── evaluate.py           # retrieval hit-rate + keyword coverage report
```

---

## Running it locally

### 1. Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and add a free Groq API key from https://console.groq.com
(the app still runs and retrieves correctly without one — it just can't
compose the final natural-language answer, and will tell you so).

```powershell
uvicorn app.main:app --reload
```

Backend is now live at `http://localhost:8000` — interactive API docs at
`http://localhost:8000/docs`.

Try it immediately with the two sample documents already in `backend/data/`:

```powershell
# from the backend/ folder, in a second terminal
curl -F "file=@data/placement_policy.txt" http://localhost:8000/ingest
curl -F "file=@data/placement_stats.txt" http://localhost:8000/ingest
```

### 2. Frontend

```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open `http://localhost:5173` — upload a document (or use the ones you just
ingested via curl above) and start asking questions.

### 3. Run the tests

```powershell
cd backend
python -m pytest tests/ -v
```

13 tests covering chunking edge cases (empty input, overlap boundaries,
no word ever silently dropped) and the full API (ingest → query → delete,
plus error handling for bad file types and empty questions). The test suite
stubs the embedding model so it runs in seconds with no network call and no
model download — useful to know and to be able to explain in an interview.

### 4. Run the evaluation harness

```powershell
cd backend
python ..\evaluation\evaluate.py
```

This is the part that separates this project from "I called a vector
search API." It runs a fixed set of questions through the live retrieval
pipeline and reports:

- **Source Hit Rate@K** — did the correct document show up in the top-K results?
- **Keyword Coverage@K** — did the retrieved text actually contain the facts needed to answer?

Swap in your own `evaluation/eval_set.json` once you've ingested your own
documents, to get a real number for your own use case.

---

## Deployment

### Backend → Render

1. Push this repo to GitHub (see below).
2. On [Render](https://render.com), create a **New Web Service**, connect
   the repo, and set the root directory to `backend`.
3. Render will detect the `Dockerfile` automatically. If it doesn't, set
   the build command to `pip install -r requirements.txt` and the start
   command to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. Add environment variables from `.env.example` in Render's dashboard —
   at minimum `GROQ_API_KEY`.
5. **Important:** Render's free-tier disk is not guaranteed to persist
   across restarts/redeploys, and local-mode Qdrant writes to disk. For a
   real deployment, spin up a free [Qdrant Cloud](https://cloud.qdrant.io)
   cluster and set `QDRANT_URL` / `QDRANT_API_KEY` in Render — the code
   switches to it automatically with no other changes (see `vectorstore.py`).

### Frontend → Vercel

1. Import the repo into [Vercel](https://vercel.com), set the root
   directory to `frontend`.
2. Add environment variable `VITE_API_URL` pointing at your deployed
   Render backend URL (e.g. `https://documind-api.onrender.com`).
3. Deploy — Vercel auto-detects the Vite build.

---

## Pushing to GitHub

From the project root, in PowerShell:

```powershell
git init
git add .
git commit -m "Initial commit: DocuMind RAG Q&A system"
git branch -M main
git remote add origin https://github.com/<your-username>/documind-rag.git
git push -u origin main
```

If you don't have a repo yet, create one first at https://github.com/new
(don't initialize it with a README, or the push above will conflict — pull
with `git pull origin main --allow-unrelated-histories` if it does).

---

## Talking points for interviews

- **Chunking:** word-based sliding window with overlap, so a fact sitting
  on a chunk boundary still appears whole in at least one chunk. Tested
  explicitly in `tests/test_chunking.py`.
- **Why local embeddings, not an API:** zero marginal cost per document,
  no rate limits during development, and it's a fair trade-off to explain —
  slightly lower quality than a hosted model like OpenAI's, in exchange for
  being free and self-contained.
- **Why Qdrant in embedded mode locally, cloud in production:** same
  client code either way — `vectorstore.py` picks local vs. remote based
  on whether `QDRANT_URL` is set. One less thing to configure to get the
  project running, without painting yourself into a corner for deployment.
- **Grounding / hallucination mitigation:** the system prompt in
  `generation.py` explicitly instructs the model to answer only from
  retrieved context and say so when the context doesn't cover the question.
- **Evaluation:** most fresher RAG projects have no way to say whether
  retrieval actually works. This one measures Source Hit Rate@K and
  Keyword Coverage@K against a fixed question set — a real, if simple,
  quality signal.
