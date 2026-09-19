from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.config import get_settings
from app.graph import build_graph, open_checkpointer, warmup_agents
from app.jobs import JobStatus, JobStore
from app.observability import init_observability
from app.worker import WorkItem, Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    init_observability()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    await redis.ping()
    await run_in_threadpool(warmup_agents)
    async with open_checkpointer(settings) as checkpointer:
        graph = build_graph(checkpointer)
        store = JobStore(redis, settings.job_ttl_seconds)
        worker = Worker(graph, store, settings)
        worker.start()
        app.state.redis, app.state.store, app.state.worker = redis, store, worker
        try:
            yield
        finally:
            await worker.stop()
            await redis.aclose()


app = FastAPI(
    title="Intelligence System API",
    version="1.0.0",
    lifespan=lifespan,
)


class TaskIn(BaseModel):
    task: str = Field(min_length=5, description="Consulta para el sistema multi-agente.")


class TaskAccepted(BaseModel):
    job_id: str
    status: JobStatus
    poll_url: str


class ApprovalIn(BaseModel):
    approved: bool
    comment: str = ""


class JobOut(BaseModel):
    job_id: str
    task: str
    status: JobStatus
    created_at: str
    updated_at: str
    error: str | None = None
    approval_request: dict[str, Any] | None = None
    approval_decision: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class ApprovalAccepted(BaseModel):
    job_id: str
    status: JobStatus
    approval_decision: dict[str, Any]


class HealthOut(BaseModel):
    redis: str
    queue_size: int


@app.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    pong = await request.app.state.redis.ping()
    return HealthOut(redis="ok" if pong else "down", queue_size=request.app.state.worker.queue_size())


@app.post("/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TaskAccepted)
async def create_task(body: TaskIn, request: Request) -> TaskAccepted:
    job_id = uuid.uuid4().hex
    await request.app.state.store.create(job_id, body.task)
    await request.app.state.worker.enqueue(WorkItem("start", job_id, {"task": body.task}))
    return TaskAccepted(job_id=job_id, status=JobStatus.PENDING, poll_url=f"/tasks/{job_id}")


@app.get("/tasks/{job_id}", response_model=JobOut)
async def get_task(job_id: str, request: Request) -> JobOut:
    job = await request.app.state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job no encontrado")
    return JobOut.model_validate(job)


@app.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED, response_model=ApprovalAccepted)
async def approve_task(job_id: str, body: ApprovalIn, request: Request) -> ApprovalAccepted:
    store: JobStore = request.app.state.store
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job no encontrado")
    if job["status"] != JobStatus.WAITING_APPROVAL:
        raise HTTPException(status_code=409, detail=f"el job está en {job['status']}, no espera aprobación")
    decision = {"approved": body.approved, "comment": body.comment}
    await store.set_status(job_id, JobStatus.PENDING, approval_decision=decision)
    await request.app.state.worker.enqueue(WorkItem("resume", job_id, decision))
    return ApprovalAccepted(job_id=job_id, status=JobStatus.PENDING, approval_decision=decision)
