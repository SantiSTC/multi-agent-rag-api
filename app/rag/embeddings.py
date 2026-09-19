from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=get_settings().embedding_model,
        encode_kwargs={"prompt": "passage: "},
        query_encode_kwargs={"prompt": "query: "},
    )
