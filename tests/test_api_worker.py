from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from redis.asyncio import Redis

import app.graph as graph_module
from app.config import Settings, get_settings
from app.jobs import JobStatus, JobStore
from app.main import app
from app.worker import WorkItem, Worker
from tests.conftest import REDIS_URL


@pytest.fixture
def dev_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = Settings(REDIS_URL=REDIS_URL, CHECKPOINTER_BACKEND="memory", WORKER_CONCURRENCY=2, JOB_TTL_SECONDS=60)
    get_settings.cache_clear()
    monkeypatch.setattr("app.main.get_settings", lambda: s)
    return s


async def wait_status(client: httpx.AsyncClient, job_id: str, *targets: str, timeout: float = 10) -> dict[str, Any]:
    for _ in range(int(timeout / 0.05)):
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] in targets:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} no llegó a {targets}: {job}")


async def test_api_full_flow_with_hitl(fake_nodes: None, redis_client: Redis, dev_settings: Settings) -> None:
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        r = await client.post("/tasks", json={"task": "¿Cuál es el precio hoy de X?"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        assert r.json()["status"] == "PENDING"

        job = await wait_status(client, job_id, "WAITING_APPROVAL")
        assert job["approval_request"]["action"] == "web_search"

        r2 = await client.post("/tasks/otro/approve", json={"approved": True})
        assert r2.status_code == 404

        r3 = await client.post(f"/tasks/{job_id}/approve", json={"approved": True, "comment": "dale"})
        assert r3.status_code == 202
        job = await wait_status(client, job_id, "DONE", "FAILED")
        assert job["status"] == "DONE"
        assert job["result"]["approval_decision"] == {"approved": True, "comment": "dale"}
        assert job["result"]["final_answer"].startswith("RESPUESTA")

        r4 = await client.post(f"/tasks/{job_id}/approve", json={"approved": True})
        assert r4.status_code == 409

        health = (await client.get("/health")).json()
        assert health["redis"] == "ok"

    store = JobStore(redis_client, 60)
    persisted = await store.get(job_id)
    assert persisted is not None and persisted["status"] == "DONE"


async def test_worker_marks_failed_on_exception(redis_client: Redis) -> None:
    class BrokenGraph:
        async def ainvoke(self, *_: Any, **__: Any) -> dict[str, Any]:
            raise RuntimeError("LLM caído")

    settings = Settings(REDIS_URL=REDIS_URL, CHECKPOINTER_BACKEND="memory", WORKER_CONCURRENCY=1)
    store = JobStore(redis_client, 60)
    worker = Worker(BrokenGraph(), store, settings)  # type: ignore[arg-type]
    worker.start()
    try:
        await store.create("j-fail", "tarea")
        await worker.enqueue(WorkItem("start", "j-fail", {"task": "tarea"}))
        for _ in range(100):
            job = await store.get("j-fail")
            if job and job["status"] == JobStatus.FAILED:
                break
            await asyncio.sleep(0.05)
        assert job["status"] == "FAILED"
        assert "RuntimeError: LLM caído" in job["error"]
    finally:
        await worker.stop()
