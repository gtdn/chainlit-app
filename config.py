from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    llm_base_url: str = Field(..., alias="LITE_LLM_BASE")
    llm_api_key: str = Field(..., alias="LITE_LLM_KEY")

    # --- Modèles ------------------------------------------------------------
    default_model: str = "groq/qwen/qwen3.8-27b"
    routing_model: str = "groq/openai/gpt-oss-20b"
    small_model: str = "llama3.2:latest"

    # --- Embeddings ---------------------------------------------------------
    embedding_model: str = "nomic-embed-text:137m-v1.5-fp16"

    # --- ChromaDB -----------------------------------------------------------
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "rag_collection"

    # --- RAG ----------------------------------------------------------------
    rag_n_results: int = 5
    rag_max_distance: float = 1.0
    chunk_size: int = 1000
    chunk_overlap: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
