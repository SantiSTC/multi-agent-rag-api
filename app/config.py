from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    llm_model: str = Field(default="openai/gpt-5-mini", alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    llm_fallback_models: str = Field(default="", alias="LLM_FALLBACK_MODELS")

    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default="intelligence-system", alias="PINECONE_INDEX")
    pinecone_namespace: str = Field(default="futbol", alias="PINECONE_NAMESPACE")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    embedding_model: str = Field(default="intfloat/multilingual-e5-small", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    chunks_file: str = Field(default="data/chunks.json", alias="CHUNKS_FILE")
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_bm25_weight: float = Field(default=0.3, alias="RAG_BM25_WEIGHT")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    checkpointer_backend: str = Field(default="redis", alias="CHECKPOINTER_BACKEND")  # redis | memory
    job_ttl_seconds: int = Field(default=86_400, alias="JOB_TTL_SECONDS")
    worker_concurrency: int = Field(default=5, alias="WORKER_CONCURRENCY")

    max_steps: int = Field(default=6, alias="MAX_STEPS")
    max_validation_rounds: int = Field(default=2, alias="MAX_VALIDATION_ROUNDS")
    recursion_limit: int = Field(default=40, alias="RECURSION_LIMIT")

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(default="intelligence-system", alias="LANGSMITH_PROJECT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    s = get_settings()
    fallbacks = [m.strip() for m in s.llm_fallback_models.split(",") if m.strip()]
    return ChatOpenAI(
        model=s.llm_model,
        api_key=s.openrouter_api_key or "missing",
        base_url=s.openrouter_base_url,
        max_tokens=s.llm_max_tokens,
        max_retries=6,
        default_headers={"X-OpenRouter-Title": "intelligence-system"},
        extra_body={"models": [s.llm_model, *fallbacks]} if fallbacks else None,
    )
