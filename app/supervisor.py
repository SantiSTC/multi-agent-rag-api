from __future__ import annotations

import re
from typing import Any, Literal, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from app.config import get_llm, get_settings
from app.state import ApprovalRequest, OrchestratorState, Route, Validation, digest

T = TypeVar("T", bound=BaseModel)

SUPERVISOR_SYSTEM = """Sos el SUPERVISOR de un sistema multi-agente jerárquico.

Dominio: FÚTBOL. La base de conocimiento interna cubre reglamento (fuera de juego, VAR),
Copa Libertadores, formaciones tácticas y los mundiales de la selección argentina.

Equipo a tu cargo:
- `research_agent`: recupera evidencia (base de conocimiento interna + web). NO calcula.
- `analyst_agent`: calcula, mide sentimiento y valida esquemas SOBRE la evidencia ya
  recolectada. NO puede buscar información nueva.
- `validator`: aplica la rúbrica de calidad y habilita (o no) la respuesta final.

Tu única decisión en cada turno: ¿quién interviene ahora?

Criterio de suficiencia (aplicalo en orden, la primera que se cumple gana):
1. Si falta evidencia o hay algún "NO ENCONTRADO" pendiente -> `research_agent`.
2. Si hay evidencia pero todavía no fue procesada (sin números, sin interpretación)
   -> `analyst_agent`.
3. Si ambos dominios ya aportaron y sus salidas cubren la pregunta -> `validator`.

Además, marcá `requires_web_search=true` SOLO si la instrucción al investigador exige
información que NO puede estar en el corpus interno (resultados recientes, fichajes,
noticias, cualquier dato posterior a los documentos). La búsqueda web tiene costo y sale a internet: es una
acción crítica que requiere aprobación humana.

Reglas duras:
- Nunca mandes al mismo agente dos veces seguidas por el mismo motivo.
- El presupuesto total es de {max_steps} intervenciones de especialistas. Si estás cerca
  del límite, mandá a `validator`.
- Si "BÚSQUEDA WEB HABILITADA: False", el humano ya rechazó la web: no vuelvas a pedirla.
- La `instruction` es lo ÚNICO que verá el especialista: autocontenida, concreta, accionable."""

_WEB_HINTS = re.compile(
    r"\b(web|internet|online|actualizad[oa]s?|últim[oa]s?|reciente|hoy|ayer|noticias?|fichajes?|"
    r"esta temporada|202[5-9])\b",
    re.IGNORECASE,
)


class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Por qué elegís ese destino, en una o dos frases.")
    next_agent: Literal["research_agent", "analyst_agent", "validator"] = Field(
        description="Nodo destino."
    )
    instruction: str = Field(description="Instrucción autocontenida y acotada para el destino.")
    requires_web_search: bool = Field(
        default=False,
        description="True solo si el investigador necesitará la herramienta web_search (acción con costo).",
    )


def is_critical(state: OrchestratorState, decision: SupervisorDecision) -> bool:
    if decision.next_agent != "research_agent":
        return False
    if state.get("approval_decision"):
        return False
    if not state.get("web_search_allowed", True):
        return False
    return decision.requires_web_search or bool(
        _WEB_HINTS.search(f"{state.get('task', '')} {decision.instruction}")
    )


async def _structured(schema: type[T], messages: list[Any], attempts: int = 2) -> T | None:
    llm = get_llm().with_structured_output(schema, method="function_calling")
    for _ in range(attempts):
        try:
            result = await llm.ainvoke(messages)
        except Exception as exc:
            if "tool_use_failed" not in str(exc) and "did not call a tool" not in str(exc):
                raise
            result = None
        if result is not None:
            return result
        messages = [*messages, HumanMessage(content="Respondé únicamente llamando a la función indicada.")]
    return None


def _fallback_decision(state: OrchestratorState) -> SupervisorDecision:
    if not state.get("research_notes"):
        return SupervisorDecision(reasoning="fallback: falta evidencia", next_agent="research_agent",
                                  instruction="Recuperá la evidencia necesaria para responder la tarea del usuario.")
    if not state.get("analyst_notes"):
        return SupervisorDecision(reasoning="fallback: falta análisis", next_agent="analyst_agent",
                                  instruction="Analizá la evidencia disponible con las herramientas de cálculo y sentimiento.")
    return SupervisorDecision(reasoning="fallback: ambos dominios aportaron", next_agent="validator",
                              instruction="Validá con la rúbrica lo aportado.")


async def supervisor_node(state: OrchestratorState) -> dict[str, Any]:
    s = get_settings()
    steps = state.get("step_count", 0)

    if steps >= s.max_steps:
        return {
            "next_agent": "validator",
            "instruction": "Presupuesto de pasos agotado: validá con lo disponible y cerrá.",
            "supervisor_notes": f"Corte por límite de pasos ({steps}/{s.max_steps}).",
            "messages": [AIMessage(content=f"[supervisor] límite de {s.max_steps} pasos -> validator", name="supervisor")],
        }

    decision = await _structured(
        SupervisorDecision,
        [SystemMessage(content=SUPERVISOR_SYSTEM.format(max_steps=s.max_steps)),
         HumanMessage(content=digest(state))],
    )
    if decision is None:
        decision = _fallback_decision(state)

    update: dict[str, Any] = {
        "next_agent": decision.next_agent,
        "instruction": decision.instruction,
        "supervisor_notes": decision.reasoning,
        "messages": [AIMessage(content=f"[supervisor -> {decision.next_agent}] {decision.reasoning}", name="supervisor")],
    }

    if is_critical(state, decision):
        request: ApprovalRequest = {
            "action": "web_search",
            "reason": "Búsqueda web externa (Tavily): herramienta con costo y salida a internet.",
            "instruction": decision.instruction,
            "target_agent": decision.next_agent,
        }
        update["next_agent"] = "human_approval"
        update["approval_request"] = request
        update["messages"].append(
            AIMessage(content="[supervisor] acción crítica detectada -> human_approval", name="supervisor")
        )
    return update


def route_from_supervisor(state: OrchestratorState) -> Route:
    nxt = state.get("next_agent")
    if nxt in ("research_agent", "analyst_agent", "validator", "human_approval"):
        return nxt  # type: ignore[return-value]
    return "validator"


VALIDATOR_SYSTEM = """Sos el VALIDADOR de calidad del sistema. No producís contenido nuevo:
solo auditás lo que trajeron los especialistas.

RÚBRICA (0 a 5 cada criterio):
- evidencia: ¿hay hallazgos concretos y citados de una fuente real?
- computo: ¿el análisis incluye números derivados con herramienta, no estimados a ojo?
- cobertura: ¿se responde TODO lo que pidió el usuario?
- consistencia: ¿los números del analista se corresponden con la evidencia del investigador?

APROBADO solo si los cuatro criterios son >= 3 y ninguno es 0.
Si rechazás, el `feedback` dice exactamente qué agente debe corregir qué cosa, en una acción."""


class ValidationVerdict(BaseModel):
    evidencia: int = Field(description="0 a 5")
    computo: int = Field(description="0 a 5")
    cobertura: int = Field(description="0 a 5")
    consistencia: int = Field(description="0 a 5")
    approved: bool
    issues: list[str] = Field(default_factory=list)
    feedback: str = ""

    @field_validator("evidencia", "computo", "cobertura", "consistencia")
    @classmethod
    def _score_ok(cls, v: int) -> int:
        return min(max(v, 0), 5)


async def validator_node(state: OrchestratorState) -> dict[str, Any]:
    verdict = await _structured(
        ValidationVerdict,
        [SystemMessage(content=VALIDATOR_SYSTEM), HumanMessage(content=digest(state))],
    )
    if verdict is None:
        has_r, has_a = bool(state.get("research_notes")), bool(state.get("analyst_notes"))
        verdict = ValidationVerdict(
            evidencia=3 if has_r else 0, computo=3 if has_a else 0,
            cobertura=3 if has_r and has_a else 1, consistencia=3 if has_r and has_a else 1,
            approved=has_r and has_a,
            issues=[] if has_r and has_a else ["El validador no devolvió veredicto estructurado."],
            feedback="" if has_r and has_a else "Completar el aporte faltante.",
        )
    scores = {
        "evidencia": verdict.evidencia, "computo": verdict.computo,
        "cobertura": verdict.cobertura, "consistencia": verdict.consistencia,
    }
    approved = all(v >= 3 for v in scores.values())
    validation: Validation = {
        "approved": approved, "scores": scores, "issues": verdict.issues, "feedback": verdict.feedback,
    }
    return {
        "validation": validation,
        "validation_rounds": state.get("validation_rounds", 0) + 1,
        "task_completed": approved,
        "messages": [AIMessage(content=f"[validator] aprobado={approved} scores={scores}", name="validator")],
    }


def route_from_validator(state: OrchestratorState) -> Literal["supervisor", "synthesizer"]:
    s = get_settings()
    validation = state.get("validation") or {}
    if validation.get("approved"):
        return "synthesizer"
    if state.get("validation_rounds", 0) >= s.max_validation_rounds:
        return "synthesizer"
    if state.get("step_count", 0) >= s.max_steps:
        return "synthesizer"
    return "supervisor"


SYNTHESIZER_SYSTEM = """Sos el redactor final del sistema multi-agente.

Combiná el aporte del investigador y el del analista en una respuesta única:
1. Respuesta directa (2 a 4 frases).
2. "Evidencia" con las citas del investigador.
3. "Análisis" con los números del analista.
4. "Limitaciones" solo si la validación quedó sin aprobar, falta algún dato o el humano
   rechazó la búsqueda web.

No agregues información que no esté en los aportes. No inventes citas ni números."""


async def synthesizer_node(state: OrchestratorState) -> dict[str, Any]:
    validation = state.get("validation") or {}
    context = digest(state)
    if not validation.get("approved", False):
        context += "\n\nATENCIÓN: la validación no fue aprobada. Incluí la sección 'Limitaciones'."
    answer = await get_llm().ainvoke(
        [SystemMessage(content=SYNTHESIZER_SYSTEM), HumanMessage(content=context)]
    )
    text = answer.text
    return {
        "final_answer": text,
        "task_completed": True,
        "messages": [AIMessage(content=text, name="synthesizer")],
    }
