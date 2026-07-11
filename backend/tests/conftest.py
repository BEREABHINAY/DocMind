"""
Stubs sentence-transformers before anything imports it, so the test
suite runs without downloading real model weights. This keeps CI fast
and network-independent while still exercising the real FastAPI
routes, real chunking logic, and a real (local, in-memory) Qdrant
instance end to end.
"""
import sys
import types
import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="session")
def _stub_sentence_transformers():
    fake_module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            vectors = []
            for text in texts:
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                v = rng.standard_normal(384).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-8)
                vectors.append(v)
            return np.array(vectors)

    fake_module.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_module
    yield
