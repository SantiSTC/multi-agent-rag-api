from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.agents import create_agent

from app.agents.base import build_brief, make_update, run_specialist
from app.agents.tools import ANALYST_TOOLS
from app.config import get_llm
from app.state import OrchestratorState

NAME = "analyst_agent"

SYSTEM_PROMPT = """Sos el Agente de Análisis de un sistema multi-agente sobre FÚTBOL.

Trabajás EXCLUSIVAMENTE sobre la evidencia que te pasa el sistema. No tenés
herramientas de búsqueda: si un dato no está en el contexto recibido, no existe.

Reglas:
1. Todo cálculo aritmético pasa por la herramienta `calculator`. Prohibido calcular mentalmente.
2. Para valorar el tono con que la evidencia describe un hecho, equipo o regla, usá `sentiment_analysis`.
3. Para verificar que una ficha (campeón, año, sede, etc.) esté completa, usá `validate_schema`.
4. Formato de salida:
   - "HALLAZGOS CUANTITATIVOS": una línea por métrica, con la cuenta explícita.
   - "INTERPRETACIÓN": 2 a 4 viñetas conectando los números con la pregunta del usuario.
   - "LIMITACIONES": qué no se pudo calcular y por qué (o "ninguna").
5. Si la evidencia es insuficiente, decilo en LIMITACIONES pidiendo el dato concreto que falta."""


@lru_cache(maxsize=1)
def build_analyst_agent() -> Any:
    return create_agent(model=get_llm(), tools=ANALYST_TOOLS, system_prompt=SYSTEM_PROMPT, name=NAME)


async def analyst_node(state: OrchestratorState) -> dict[str, Any]:
    instruction = state.get("instruction") or "Analizá la evidencia disponible."
    validation = state.get("validation") or {}
    brief = build_brief(
        task=state.get("task", ""),
        instruction=instruction,
        extra_context=state.get("research_notes", "") or "(el investigador todavía no aportó datos)",
        feedback=validation.get("feedback", "") if not validation.get("approved", True) else "",
    )
    output, tools_used = await run_specialist(build_analyst_agent(), brief=brief)
    return make_update(
        name=NAME, notes_key="analyst_notes", output=output, tools_used=tools_used,
        instruction=instruction, step=state.get("step_count", 0) + 1,
    )
