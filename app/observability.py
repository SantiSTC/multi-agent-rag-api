from __future__ import annotations

import logging
import os
from typing import Any

from langsmith import traceable

log = logging.getLogger("observability")


def init_observability() -> bool:
    enabled = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
    if enabled and not os.getenv("LANGSMITH_API_KEY"):
        log.warning("LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: no se enviarán trazas.")
        return False
    if enabled:
        log.info("LangSmith activo -> proyecto '%s'", os.getenv("LANGSMITH_PROJECT", "default"))
    else:
        log.info("LangSmith desactivado (LANGSMITH_TRACING no es 'true').")
    return enabled


def run_config(job_id: str, *, recursion_limit: int, extra_tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": job_id},
        "recursion_limit": recursion_limit,
        "run_name": "orchestrator_job",
        "tags": ["orchestrator", *(extra_tags or [])],
        "metadata": {"job_id": job_id, "thread_id": job_id},
    }


__all__ = ["init_observability", "run_config", "traceable"]
