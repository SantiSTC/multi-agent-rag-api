from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.state import Contribution


def build_brief(*, task: str, instruction: str, extra_context: str = "", feedback: str = "") -> str:
    parts = [f"OBJETIVO GENERAL DEL USUARIO:\n{task}", f"\nTU TAREA CONCRETA AHORA:\n{instruction}"]
    if extra_context.strip():
        parts.append(f"\nCONTEXTO YA PRODUCIDO POR OTRO ESPECIALISTA (no lo repitas):\n{extra_context}")
    if feedback.strip():
        parts.append(f"\nCORRECCIÓN PEDIDA POR EL VALIDADOR (resolvela explícitamente):\n{feedback}")
    return "\n".join(parts)


async def run_specialist(agent: Any, *, brief: str, recursion_limit: int = 25) -> tuple[str, list[str]]:
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=brief)]},
        config={"recursion_limit": recursion_limit},
    )
    messages = result["messages"]
    tools_used: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            tools_used.extend(call["name"] for call in msg.tool_calls or [])
        elif isinstance(msg, ToolMessage) and msg.name and msg.name not in tools_used:
            tools_used.append(msg.name)
    return messages[-1].text.strip(), tools_used


def make_update(
    *, name: str, notes_key: str, output: str, tools_used: list[str], instruction: str, step: int
) -> dict[str, Any]:
    contribution: Contribution = {
        "step": step, "agent": name, "instruction": instruction, "tools_used": tools_used, "output": output,
    }
    return {
        notes_key: output,
        "contributions": [contribution],
        "step_count": step,
        "messages": [AIMessage(content=output, name=name)],
    }
