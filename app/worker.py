from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.config import Settings
from app.graph import initial_state
from app.jobs import JobStatus, JobStore
from app.observability import run_config, traceable

log = logging.getLogger("worker")


@dataclass(slots=True)
class WorkItem:
    kind: Literal["start", "resume"]
    job_id: str
    payload: dict[str, Any]


class Worker:
    def __init__(self, graph: CompiledStateGraph, store: JobStore, settings: Settings) -> None:
        self._graph = graph
        self._store = store
        self._settings = settings
        self._queue: asyncio.Queue[WorkItem] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._loop(i), name=f"worker-{i}")
            for i in range(self._settings.worker_concurrency)
        ]
        log.info("worker: %d consumidores arrancados", len(self._tasks))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def enqueue(self, item: WorkItem) -> None:
        await self._queue.put(item)

    def queue_size(self) -> int:
        return self._queue.qsize()

    async def _loop(self, idx: int) -> None:
        while True:
            item = await self._queue.get()
            try:
                await self._process(item)
            except Exception:
                log.exception("worker-%d: error no manejado en job %s", idx, item.job_id)
            finally:
                self._queue.task_done()

    async def _process(self, item: WorkItem) -> None:
        job_id = item.job_id
        await self._store.set_status(job_id, JobStatus.RUNNING)
        try:
            if item.kind == "start":
                result = await self._run_graph(job_id, initial_state(item.payload["task"]), kind="start")
            else:
                result = await self._run_graph(job_id, Command(resume=item.payload), kind="resume")
        except Exception as exc:
            log.exception("job %s FAILED", job_id)
            await self._store.set_status(job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            return

        if "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0].value
            await self._store.set_status(job_id, JobStatus.WAITING_APPROVAL, approval_request=interrupt_payload)
            log.info("job %s WAITING_APPROVAL", job_id)
            return

        await self._store.set_status(
            job_id,
            JobStatus.DONE,
            result={
                "final_answer": result.get("final_answer", ""),
                "validation": result.get("validation"),
                "approval_decision": result.get("approval_decision"),
                "contributions": [
                    {"step": c["step"], "agent": c["agent"], "tools_used": c["tools_used"]}
                    for c in result.get("contributions", [])
                ],
                "steps": result.get("step_count", 0),
            },
        )
        log.info("job %s DONE", job_id)

    async def _run_graph(self, job_id: str, graph_input: Any, *, kind: str) -> dict[str, Any]:
        return await run_job(
            self._graph, job_id, graph_input, kind=kind, recursion_limit=self._settings.recursion_limit
        )


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"job_id": inputs.get("job_id"), "kind": inputs.get("kind")}


def _trace_outputs(outputs: Any) -> dict[str, Any]:
    if not isinstance(outputs, dict):
        return {"result": str(outputs)}
    return {
        "interrupted": "__interrupt__" in outputs,
        "steps": outputs.get("step_count"),
        "final_answer": outputs.get("final_answer", ""),
    }


@traceable(
    name="orchestrator_job",
    run_type="chain",
    tags=["worker"],
    process_inputs=_trace_inputs,
    process_outputs=_trace_outputs,
)
async def run_job(
    graph: CompiledStateGraph, job_id: str, graph_input: Any, *, kind: str, recursion_limit: int
) -> dict[str, Any]:
    config = run_config(job_id, recursion_limit=recursion_limit, extra_tags=[kind])
    return await graph.ainvoke(graph_input, config=config)
