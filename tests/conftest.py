from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from redis.asyncio import Redis

import app.graph as graph_module
from app.state import OrchestratorState

REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")


async def fake_supervisor(state: OrchestratorState) -> dict[str, Any]:
    if not state.get("research_notes"):
        if not state.get("approval_decision"):
            return {
                "next_agent": "human_approval",
                "instruction": "Buscá en la web el precio actual de X.",
                "supervisor_notes": "hace falta info externa",
                "approval_request": {
                    "action": "web_search", "reason": "costo", "instruction": "Buscá en la web el precio actual de X.",
                    "target_agent": "research_agent",
                },
                "messages": [AIMessage(content="[supervisor] -> human_approval", name="supervisor")],
            }
        return {"next_agent": "research_agent", "instruction": "buscar", "supervisor_notes": "", "messages": []}
    if not state.get("analyst_notes"):
        return {"next_agent": "analyst_agent", "instruction": "calcular", "supervisor_notes": "", "messages": []}
    return {"next_agent": "validator", "instruction": "validar", "supervisor_notes": "", "messages": []}


async def fake_research(state: OrchestratorState) -> dict[str, Any]:
    web = state.get("web_search_allowed", True)
    return {
        "research_notes": f"evidencia (web={web})",
        "contributions": [{"step": 1, "agent": "research_agent", "instruction": "", "tools_used": ["knowledge_base_search"] + (["web_search"] if web else []), "output": "e"}],
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content="evidencia", name="research_agent")],
    }


async def fake_analyst(state: OrchestratorState) -> dict[str, Any]:
    return {
        "analyst_notes": "30.6%",
        "contributions": [{"step": 2, "agent": "analyst_agent", "instruction": "", "tools_used": ["calculator"], "output": "a"}],
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content="analisis", name="analyst_agent")],
    }


async def fake_validator(state: OrchestratorState) -> dict[str, Any]:
    return {
        "validation": {"approved": True, "scores": {"evidencia": 4, "computo": 4, "cobertura": 4, "consistencia": 4}, "issues": [], "feedback": ""},
        "validation_rounds": state.get("validation_rounds", 0) + 1,
        "task_completed": True,
        "messages": [],
    }


async def fake_synthesizer(state: OrchestratorState) -> dict[str, Any]:
    return {"final_answer": f"RESPUESTA ({state['research_notes']})", "task_completed": True, "messages": []}


@pytest.fixture
def fake_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "supervisor_node", fake_supervisor)
    monkeypatch.setattr(graph_module, "research_node", fake_research)
    monkeypatch.setattr(graph_module, "analyst_node", fake_analyst)
    monkeypatch.setattr(graph_module, "validator_node", fake_validator)
    monkeypatch.setattr(graph_module, "synthesizer_node", fake_synthesizer)
    monkeypatch.setattr("app.main.warmup_agents", lambda: None)


@pytest.fixture
async def redis_client() -> Redis:
    r = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await r.ping()
    except Exception:
        pytest.skip(f"Redis no disponible en {REDIS_URL}")
    await r.flushdb()
    yield r
    await r.aclose()
