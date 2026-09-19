from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.state import ApprovalDecision, OrchestratorState, Route


def human_approval_node(state: OrchestratorState) -> dict[str, Any]:
    request = state.get("approval_request") or {}

    raw: Any = interrupt({
        "type": "approval_required",
        "action": request.get("action", ""),
        "reason": request.get("reason", ""),
        "instruction": request.get("instruction", ""),
        "target_agent": request.get("target_agent", ""),
        "how_to_resume": "POST /tasks/{job_id}/approve  body: {\"approved\": true|false, \"comment\": \"...\"}",
    })

    approved = bool(raw.get("approved", False)) if isinstance(raw, dict) else bool(raw)
    comment = str(raw.get("comment", "")) if isinstance(raw, dict) else ""
    decision: ApprovalDecision = {"approved": approved, "comment": comment}

    verdict = "APROBADA" if approved else "RECHAZADA (se sigue solo con el corpus interno)"
    return {
        "approval_decision": decision,
        "web_search_allowed": approved,
        "next_agent": request.get("target_agent", "research_agent"),
        "messages": [AIMessage(content=f"[human_approval] acción '{request.get('action')}' {verdict}. {comment}".strip(),
                               name="human_approval")],
    }


def route_from_approval(state: OrchestratorState) -> Route:
    nxt = state.get("next_agent")
    return nxt if nxt in ("research_agent", "analyst_agent", "validator") else "research_agent"  # type: ignore[return-value]
