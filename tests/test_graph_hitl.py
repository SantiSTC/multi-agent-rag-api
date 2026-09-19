from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import build_graph, initial_state
from app.supervisor import SupervisorDecision, is_critical, route_from_supervisor


def test_graph_structure() -> None:
    g = build_graph().get_graph()
    assert {"supervisor", "research_agent", "analyst_agent", "validator", "synthesizer", "human_approval"} <= set(g.nodes)
    edges = {(e.source, e.target) for e in g.edges}
    assert ("supervisor", "human_approval") in edges
    assert ("human_approval", "research_agent") in edges
    assert ("research_agent", "supervisor") in edges
    assert ("synthesizer", "__end__") in edges


def test_is_critical_rule() -> None:
    web = SupervisorDecision(reasoning="", next_agent="research_agent", instruction="buscar precios actuales", requires_web_search=True)
    kb = SupervisorDecision(reasoning="", next_agent="research_agent", instruction="buscar chunking en el corpus")
    assert is_critical({"task": "x"}, web) is True
    assert is_critical({"task": "¿cuál es el precio hoy?"}, kb) is True
    assert is_critical({"task": "qué es RAG"}, kb) is False
    assert is_critical({"task": "x", "approval_decision": {"approved": False, "comment": ""}}, web) is False
    assert is_critical({"task": "x", "web_search_allowed": False}, web) is False
    assert route_from_supervisor({"next_agent": "human_approval"}) == "human_approval"
    assert route_from_supervisor({}) == "validator"


async def test_hitl_pauses_and_resumes_with_rejection(fake_nodes: None) -> None:
    graph = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "job-1"}, "recursion_limit": 40}

    first = await graph.ainvoke(initial_state("¿Cuál es el precio hoy de X?"), cfg)
    assert "__interrupt__" in first
    payload = first["__interrupt__"][0].value
    assert payload["type"] == "approval_required" and payload["action"] == "web_search"

    snapshot = await graph.aget_state(cfg)
    assert snapshot.next == ("human_approval",)

    final = await graph.ainvoke(Command(resume={"approved": False, "comment": "sin presupuesto"}), cfg)
    assert "__interrupt__" not in final
    assert final["approval_decision"] == {"approved": False, "comment": "sin presupuesto"}
    assert final["web_search_allowed"] is False
    assert "web=False" in final["research_notes"]
    assert final["final_answer"].startswith("RESPUESTA")


async def test_hitl_resume_with_approval(fake_nodes: None) -> None:
    graph = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "job-2"}, "recursion_limit": 40}
    await graph.ainvoke(initial_state("precio actual de X"), cfg)
    final = await graph.ainvoke(Command(resume={"approved": True, "comment": "ok"}), cfg)
    assert final["web_search_allowed"] is True
    assert "web_search" in final["contributions"][0]["tools_used"]
