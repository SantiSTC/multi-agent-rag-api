from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

Route = Literal["research_agent", "analyst_agent", "validator", "human_approval"]


class Contribution(TypedDict):
    step: int
    agent: str
    instruction: str
    tools_used: list[str]
    output: str


class Validation(TypedDict):
    approved: bool
    scores: dict[str, int]
    issues: list[str]
    feedback: str


class ApprovalRequest(TypedDict):
    action: str
    reason: str
    instruction: str
    target_agent: str


class ApprovalDecision(TypedDict):
    approved: bool
    comment: str


class OrchestratorState(MessagesState):
    task: str
    next_agent: str
    instruction: str
    supervisor_notes: str
    approval_request: ApprovalRequest
    approval_decision: ApprovalDecision
    web_search_allowed: bool
    research_notes: str
    analyst_notes: str
    contributions: Annotated[list[Contribution], operator.add]
    validation: Validation
    step_count: int
    validation_rounds: int
    task_completed: bool
    final_answer: str


def digest(state: OrchestratorState) -> str:
    research = state.get("research_notes") or "(sin datos todavía)"
    analysis = state.get("analyst_notes") or "(sin análisis todavía)"
    validation = state.get("validation")
    decision = state.get("approval_decision")

    lines = [
        f"TAREA DEL USUARIO: {state.get('task', '')}",
        f"PASOS CONSUMIDOS: {state.get('step_count', 0)}",
        f"BÚSQUEDA WEB HABILITADA: {state.get('web_search_allowed', True)}",
        "",
        "--- APORTE DEL AGENTE DE INVESTIGACIÓN ---",
        research,
        "",
        "--- APORTE DEL AGENTE DE ANÁLISIS ---",
        analysis,
    ]
    if decision:
        lines += ["", f"--- DECISIÓN HUMANA SOBRE LA ACCIÓN CRÍTICA --- aprobada={decision['approved']} "
                  f"comentario: {decision.get('comment', '')}"]
    if validation:
        lines += [
            "",
            "--- ÚLTIMO VEREDICTO DEL VALIDADOR ---",
            f"aprobado={validation.get('approved')} | scores={validation.get('scores')}",
            f"faltantes: {'; '.join(validation.get('issues') or []) or 'ninguno'}",
            f"feedback: {validation.get('feedback', '')}",
        ]
    return "\n".join(lines)
