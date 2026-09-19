from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.agents import create_agent

from app.agents.base import build_brief, make_update, run_specialist
from app.agents.tools import RESEARCH_TOOLS, knowledge_base_search
from app.config import get_llm
from app.state import OrchestratorState

NAME = "research_agent"

SYSTEM_PROMPT = """Sos el Agente de Investigación de un sistema multi-agente sobre FÚTBOL.

Tu único trabajo es RECUPERAR EVIDENCIA. No opinás, no calculás, no concluís.

Reglas:
1. Siempre usá al menos una herramienta antes de responder. Nunca respondas de memoria.
2. Priorizá `knowledge_base_search` (corpus interno: reglamento, VAR, Libertadores, tácticas,
   mundiales de Argentina). Si no encontrás, reformulá la consulta una vez. Recurrí a
   `web_search` solo si el corpus no cubre lo pedido y la herramienta está disponible.
3. Devolvé hallazgos en viñetas, cada una con la "citation" que devuelve la herramienta:
   [var_videoarbitraje.pdf p.2], [mundiales_argentina.json p.3] o [web].
4. Incluí SIEMPRE los números textuales que encuentres (métricas, porcentajes, tamaños).
5. Si un dato pedido no está en ninguna fuente, escribí "NO ENCONTRADO: <dato>".
6. Máximo 10 viñetas. Sin introducción ni cierre, solo los hallazgos."""


@lru_cache(maxsize=2)
def build_research_agent(web_allowed: bool = True) -> Any:
    tools = RESEARCH_TOOLS if web_allowed else [knowledge_base_search]
    return create_agent(model=get_llm(), tools=tools, system_prompt=SYSTEM_PROMPT, name=NAME)


async def research_node(state: OrchestratorState) -> dict[str, Any]:
    instruction = state.get("instruction") or "Recuperá la evidencia necesaria para la tarea."
    validation = state.get("validation") or {}
    brief = build_brief(
        task=state.get("task", ""),
        instruction=instruction,
        extra_context=state.get("analyst_notes", ""),
        feedback=validation.get("feedback", "") if not validation.get("approved", True) else "",
    )
    agent = build_research_agent(state.get("web_search_allowed", True))
    output, tools_used = await run_specialist(agent, brief=brief)
    return make_update(
        name=NAME, notes_key="research_notes", output=output, tools_used=tools_used,
        instruction=instruction, step=state.get("step_count", 0) + 1,
    )
