"""
Central configuration for DocuMind.

All tunables live here so the rest of the codebase never reads
os.environ directly. This makes it obvious, in one place, exactly
what the system depends on to run.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM provider (Groq is OpenAI-API-compatible, so we reuse the
    # openai SDK and just point it at Groq's base URL) ---
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.1-8b-instant"

    # --- Embeddings ---
    # Runs locally via sentence-transformers, no API key needed.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- Vector store (Qdrant, embedded/local mode by default so the
    # project runs with zero external accounts. Point qdrant_url at a
    # Qdrant Cloud instance for a real deployment.) ---
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_local_path: str = "./qdrant_data"
    collection_name: str = "documents"

    # --- Chunking ---
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40

    # --- Retrieval ---
    top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
