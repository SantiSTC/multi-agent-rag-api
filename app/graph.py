from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.analyst_agent import analyst_node
from app.agents.research_agent import research_node
from app.config import Settings, get_settings
from app.hitl import human_approval_node, route_from_approval
from app.state import OrchestratorState
from app.supervisor import (
    route_from_supervisor,
    route_from_validator,
    supervisor_node,
    synthesizer_node,
    validator_node,
)


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(OrchestratorState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research_agent", research_node)
    builder.add_node("analyst_agent", analyst_node)
    builder.add_node("validator", validator_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("human_approval", human_approval_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research_agent": "research_agent",
            "analyst_agent": "analyst_agent",
            "validator": "validator",
            "human_approval": "human_approval",
        },
    )
    builder.add_conditional_edges(
        "human_approval",
        route_from_approval,
        {"research_agent": "research_agent", "analyst_agent": "analyst_agent", "validator": "validator"},
    )
    builder.add_edge("research_agent", "supervisor")
    builder.add_edge("analyst_agent", "supervisor")
    builder.add_conditional_edges(
        "validator", route_from_validator, {"supervisor": "supervisor", "synthesizer": "synthesizer"}
    )
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer)


def warmup_agents() -> None:
    from app.agents.analyst_agent import build_analyst_agent
    from app.agents.research_agent import build_research_agent
    from app.agents.tools import get_retriever
    from app.config import get_llm

    get_llm()
    get_retriever()
    build_research_agent(True)
    build_research_agent(False)
    build_analyst_agent()


def initial_state(task: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": task}],
        "task": task,
        "next_agent": "",
        "instruction": "",
        "supervisor_notes": "",
        "web_search_allowed": True,
        "research_notes": "",
        "analyst_notes": "",
        "contributions": [],
        "step_count": 0,
        "validation_rounds": 0,
        "task_completed": False,
        "final_answer": "",
    }


@asynccontextmanager
async def open_checkpointer(settings: Settings | None = None) -> AsyncIterator[BaseCheckpointSaver]:
    settings = settings or get_settings()
    if settings.checkpointer_backend == "memory":
        yield InMemorySaver()
        return

    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(settings.redis_url) as saver:
        await saver.asetup()
        yield saver
