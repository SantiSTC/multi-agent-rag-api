from __future__ import annotations

import ast
import operator as op
import os
import re
from functools import lru_cache

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from app.rag.retriever import HybridRetriever
from app.rag.schemas import SearchQuery, SearchResult

@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


class ToolError(BaseModel):
    error: str
    sugerencia: str = ""


@tool(args_schema=SearchQuery)
async def knowledge_base_search(query: str, k: int = 5, categoria: str = "") -> str:
    """Busca fragmentos relevantes en la base de conocimiento interna sobre FÚTBOL.

    Corpus disponible (5 documentos): reglamento (regla del fuera de juego, VAR /
    videoarbitraje), torneos (Copa Libertadores), tácticas (formaciones tácticas)
    y selecciones (mundiales de Argentina). Recuperación híbrida BM25 + vectorial.

    Usar SIEMPRE esta herramienta antes que `web_search`. Formular la consulta en
    lenguaje natural y específica ("cuándo interviene el VAR", "títulos de Boca en
    la Libertadores"). Se puede filtrar con `categoria` = reglamento | torneos |
    tacticas | selecciones. Si no aparece lo buscado, reformular la consulta con
    otros términos antes de darlo por no encontrado.

    Devuelve JSON con "hits": lista de {text, source, page, categoria, score,
    citation}. Citar cada hallazgo con su "citation" (ej. [var_videoarbitraje.pdf p.2]).
    """
    try:
        result: SearchResult = await get_retriever().search(
            SearchQuery(query=query, k=k, categoria=categoria)
        )
    except Exception as exc:
        return ToolError(error=f"{type(exc).__name__}: {exc}",
                         sugerencia="Reintentar con otra consulta o sin filtro de categoría.").model_dump_json()
    if not result.hits:
        return ToolError(error="Sin resultados en la base de conocimiento.",
                         sugerencia="Reformular con otros términos o quitar el filtro de categoría.").model_dump_json()
    return result.model_dump_json()


class WebSearchInput(BaseModel):
    query: str = Field(description="Consulta a buscar en la web.")

    @field_validator("query")
    @classmethod
    def _query_ok(cls, v: str) -> str:
        if not 3 <= len(v.strip()) <= 300:
            raise ValueError("query debe tener entre 3 y 300 caracteres")
        return v.strip()


class WebSearchResult(BaseModel):
    query: str
    results: list[dict[str, str]]


@tool(args_schema=WebSearchInput)
async def web_search(query: str) -> str:
    """Busca información actualizada en la web mediante Tavily (herramienta externa con costo).

    Usar SOLO cuando la base de conocimiento interna no cubra lo pedido: datos
    posteriores al corpus (resultados recientes, noticias, fichajes actuales).
    Esta herramienta puede no estar disponible si el humano no aprobó su uso.

    Devuelve JSON con "results": lista de {title, url, content}; citar como [web].
    """
    if not os.getenv("TAVILY_API_KEY"):
        return ToolError(error="TAVILY_API_KEY no configurada: búsqueda web deshabilitada.",
                         sugerencia="Usar knowledge_base_search.").model_dump_json()
    from langchain_tavily import TavilySearch

    raw = await TavilySearch(max_results=4, topic="general").ainvoke({"query": query})
    items = raw.get("results", []) if isinstance(raw, dict) else raw
    results = [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
               for r in items]
    if not results:
        return ToolError(error="Sin resultados web.").model_dump_json()
    return WebSearchResult(query=query, results=results).model_dump_json()


RESEARCH_TOOLS = [knowledge_base_search, web_search]


_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Expresión no permitida")


class CalculatorInput(BaseModel):
    expression: str = Field(description='Expresión aritmética, ej. "3 / 18 * 100".')

    @field_validator("expression")
    @classmethod
    def _expr_ok(cls, v: str) -> str:
        if not 1 <= len(v.strip()) <= 200:
            raise ValueError("expression vacía o demasiado larga")
        return v.strip()


class CalculatorResult(BaseModel):
    expression: str
    value: float


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Evalúa una expresión aritmética de forma segura (sin eval de Python).

    Soporta + - * / // % ** y paréntesis. Ejemplo: "3 / 18 * 100" para un porcentaje.
    Todo número que aparezca en el análisis tiene que salir de esta herramienta.
    Devuelve JSON {expression, value} o {"error": ...} si la expresión no es válida.
    """
    try:
        value = _eval_node(ast.parse(expression, mode="eval").body)
    except Exception as exc:
        return ToolError(error=f"Expresión inválida '{expression}': {exc}",
                         sugerencia="Usar solo números, operadores aritméticos y paréntesis.").model_dump_json()
    return CalculatorResult(expression=expression, value=float(value)).model_dump_json()


_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)
_POSITIVE = {
    "bueno", "buena", "mejor", "excelente", "eficaz", "eficiente", "rápido", "sólido",
    "ventaja", "mejora", "aumenta", "supera", "gesta", "histórico", "histórica", "gloria",
    "campeón", "campeona", "triunfo", "victoria", "éxito", "dominante", "brillante",
}
_NEGATIVE = {
    "malo", "mala", "peor", "lento", "costoso", "riesgo", "riesgos", "falla", "fallas",
    "error", "errores", "problema", "problemas", "polémica", "polémico", "controversia",
    "derrota", "fracaso", "eliminación", "crítica", "críticas", "injusto", "lesión",
}


class SentimentInput(BaseModel):
    text: str = Field(description="Texto a analizar.")

    @field_validator("text")
    @classmethod
    def _text_ok(cls, v: str) -> str:
        if not 1 <= len(v.strip()) <= 5000:
            raise ValueError("text vacío o demasiado largo")
        return v


class SentimentResult(BaseModel):
    label: str
    score: float = Field(ge=-1, le=1)
    positive_hits: list[str]
    negative_hits: list[str]
    tokens_analizados: int


@tool(args_schema=SentimentInput)
def sentiment_analysis(text: str) -> str:
    """Analiza el tono de un texto (positivo / neutral / negativo) con un léxico determinista.

    Usar para valorar cómo se describe un hecho, un equipo o una regla en la
    evidencia (ej.: "¿el texto trata al VAR como polémico o como una mejora?").
    Devuelve JSON {label, score en [-1, 1], positive_hits, negative_hits, tokens_analizados}.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text.lower())]
    pos = [t for t in tokens if t in _POSITIVE]
    neg = [t for t in tokens if t in _NEGATIVE]
    total = len(pos) + len(neg)
    score = 0.0 if total == 0 else round((len(pos) - len(neg)) / total, 3)
    label = "positivo" if score > 0.2 else "negativo" if score < -0.2 else "neutral"
    return SentimentResult(label=label, score=score, positive_hits=sorted(set(pos)),
                           negative_hits=sorted(set(neg)), tokens_analizados=len(tokens)).model_dump_json()


class ValidateSchemaInput(BaseModel):
    data: dict[str, object] = Field(description="Objeto a validar.")
    required_fields: list[str] = Field(description="Campos obligatorios.")

    @field_validator("required_fields")
    @classmethod
    def _fields_ok(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("required_fields no puede estar vacío")
        return v


class ValidateSchemaResult(BaseModel):
    valid: bool
    missing_fields: list[str]
    present_fields: list[str]


@tool(args_schema=ValidateSchemaInput)
def validate_schema(data: dict[str, object], required_fields: list[str]) -> str:
    """Verifica que un objeto contenga todos los campos requeridos y reporta los faltantes.

    Usar para chequear que una ficha armada a partir de la evidencia esté completa
    (ej.: {"campeon": ..., "anio": ..., "sede": ...} con required_fields
    ["campeon", "anio", "sede"]). Devuelve JSON {valid, missing_fields, present_fields}.
    """
    missing = [f for f in required_fields if f not in data]
    return ValidateSchemaResult(valid=not missing, missing_fields=missing,
                                present_fields=sorted(data)).model_dump_json()


ANALYST_TOOLS = [calculator, sentiment_analysis, validate_schema]
