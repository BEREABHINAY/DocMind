import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Fresh app + fresh isolated vector store per test, so tests don't
    leak state into each other."""
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "qdrant_data"))

    # app.config / app.vectorstore are singletons created at import time,
    # so make sure every test gets a clean module import.
    for mod in list(sys.modules):
        if mod.startswith("app."):
            del sys.modules[mod]

    from app.main import app

    return TestClient(app)


def test_health_starts_empty(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "documents_indexed": 0}


def test_ingest_and_query_roundtrip(client):
    with open(DATA_DIR / "placement_policy.txt", "rb") as f:
        r = client.post("/ingest", files={"file": ("placement_policy.txt", f, "text/plain")})
    assert r.status_code == 200
    assert r.json()["chunks_created"] > 0

    r = client.get("/health")
    assert r.json()["documents_indexed"] > 0

    r = client.post("/query", json={"question": "minimum aggregate percentage"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert len(body["sources"]) > 0
    assert body["sources"][0]["source"] == "placement_policy.txt"


def test_documents_listing_and_deletion(client):
    with open(DATA_DIR / "placement_policy.txt", "rb") as f:
        client.post("/ingest", files={"file": ("placement_policy.txt", f, "text/plain")})
    with open(DATA_DIR / "placement_stats.txt", "rb") as f:
        client.post("/ingest", files={"file": ("placement_stats.txt", f, "text/plain")})

    docs = client.get("/documents").json()
    sources = {d["source"] for d in docs}
    assert sources == {"placement_policy.txt", "placement_stats.txt"}

    client.delete("/documents/placement_stats.txt")
    docs = client.get("/documents").json()
    sources = {d["source"] for d in docs}
    assert sources == {"placement_policy.txt"}


def test_rejects_unsupported_file_type(client):
    r = client.post(
        "/ingest", files={"file": ("virus.exe", b"binary-junk", "application/octet-stream")}
    )
    assert r.status_code == 400


def test_rejects_empty_upload(client):
    r = client.post("/ingest", files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code == 400


def test_rejects_empty_question(client):
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 400


def test_query_with_no_documents_indexed_yet(client):
    r = client.post("/query", json={"question": "anything at all"})
    assert r.status_code == 200
    assert r.json()["sources"] == []
