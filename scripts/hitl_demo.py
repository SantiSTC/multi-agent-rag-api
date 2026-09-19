from __future__ import annotations

import argparse
import asyncio
import json

import httpx

TASK = (
    "Buscá en la web el último campeón de la Copa Libertadores (esta temporada) y compará su "
    "cantidad de títulos con la del máximo ganador histórico según el corpus."
)


async def wait_for(client: httpx.AsyncClient, job_id: str, *targets: str, timeout: float = 600) -> dict:
    for _ in range(int(timeout)):
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] in targets:
            return job
        await asyncio.sleep(1)
    raise TimeoutError(job_id)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--reject", action="store_true")
    args = ap.parse_args()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        job_id = (await client.post("/tasks", json={"task": TASK})).json()["job_id"]
        print(f"job_id={job_id} -> encolado")

        job = await wait_for(client, job_id, "WAITING_APPROVAL", "DONE", "FAILED")
        if job["status"] != "WAITING_APPROVAL":
            print(f"El sistema no consideró crítica la tarea (status={job['status']}).")
            print(json.dumps(job.get("result") or job.get("error"), ensure_ascii=False, indent=2))
            return

        print("\n⏸  El grafo está PAUSADO esperando aprobación humana:")
        print(json.dumps(job["approval_request"], ensure_ascii=False, indent=2))

        decision = {"approved": not args.reject, "comment": "demo HITL"}
        print(f"\n→ POST /tasks/{job_id}/approve {decision}")
        await client.post(f"/tasks/{job_id}/approve", json=decision)

        job = await wait_for(client, job_id, "DONE", "FAILED")
        print(f"\nstatus={job['status']}")
        print(json.dumps(job.get("result") or job.get("error"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
