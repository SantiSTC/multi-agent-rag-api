from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    def __init__(self, redis: Redis, ttl_seconds: int) -> None:
        self._r = redis
        self._ttl = ttl_seconds

    @staticmethod
    def _key(job_id: str) -> str:
        return f"job:{job_id}"

    async def create(self, job_id: str, task: str) -> dict[str, Any]:
        job = {
            "job_id": job_id, "task": task, "status": JobStatus.PENDING,
            "created_at": _now(), "updated_at": _now(),
        }
        await self._r.hset(self._key(job_id), mapping=job)
        await self._r.expire(self._key(job_id), self._ttl)
        return job

    async def set_status(self, job_id: str, status: JobStatus, **fields: Any) -> None:
        payload: dict[str, str] = {"status": status, "updated_at": _now()}
        for k, v in fields.items():
            payload[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        await self._r.hset(self._key(job_id), mapping=payload)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        raw = await self._r.hgetall(self._key(job_id))
        if not raw:
            return None
        job: dict[str, Any] = dict(raw)
        for k in ("result", "approval_request", "approval_decision"):
            if k in job:
                try:
                    job[k] = json.loads(job[k])
                except json.JSONDecodeError:
                    pass
        return job
