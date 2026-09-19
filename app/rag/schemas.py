from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Chunk(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(description="Nombre del archivo de origen, ej. var_videoarbitraje.pdf")
    categoria: str = Field(description="reglamento | torneos | tacticas | selecciones")
    page: int = Field(ge=1)


class SearchQuery(BaseModel):
    query: str = Field(description="Consulta en lenguaje natural.")
    k: int = Field(default=5, description="Cantidad máxima de fragmentos a devolver (1 a 20).")
    categoria: str = Field(default="", description="Filtro opcional: reglamento | torneos | tacticas | selecciones.")

    @field_validator("query")
    @classmethod
    def _query_ok(cls, v: str) -> str:
        v = v.strip()
        if not 3 <= len(v) <= 500:
            raise ValueError("query debe tener entre 3 y 500 caracteres")
        return v

    @field_validator("k")
    @classmethod
    def _k_ok(cls, v: int) -> int:
        return min(max(v, 1), 20)


class SearchHit(BaseModel):
    text: str
    source: str
    categoria: str
    page: int
    score: float = Field(description="Puntaje de fusión RRF (mayor = más relevante).")
    citation: str = Field(description="Cita corta para usar en la respuesta, ej. [var_videoarbitraje.pdf p.3]")


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]
    total: int
