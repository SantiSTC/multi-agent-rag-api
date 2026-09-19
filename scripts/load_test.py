from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

TASKS = [
    "¿Cuándo puede intervenir el VAR en un partido y qué porcentaje de las cuatro situaciones "
    "revisables involucran goles? Calculalo.",
    "¿Cuántos mundiales ganó la selección argentina y en qué años? Calculá qué porcentaje "
    "de los mundiales que jugó terminó ganando.",
    "¿Qué equipo tiene más títulos de Copa Libertadores? Calculá la diferencia con el segundo "
    "y analizá el tono con que el texto describe a ese equipo.",
    "Explicá la regla del fuera de juego y analizá si el texto la presenta como polémica o "
    "como una regla clara.",
    "Compará la formación 4-3-3 con la 4-4-2 según el corpus: ¿cuántos jugadores cambian de "
    "línea? Calculalo y analizá el tono de la descripción de cada una.",
]


async def run_one(client: httpx.AsyncClient, task: str, poll_every: float, timeout: float) -> dict:
    t0 = time.perf_counter()
    r = await client.post("/tasks", json={"task": task})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    t_accept = time.perf_counter() - t0

    while time.perf_counter() - t0 < timeout:
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] in ("DONE", "FAILED"):
            return {"job_id": job_id, "status": job["status"], "accept_ms": t_accept * 1000,
                    "total_s": time.perf_counter() - t0, "error": job.get("error")}
        if job["status"] == "WAITING_APPROVAL":
            await client.post(f"/tasks/{job_id}/approve", json={"approved": True, "comment": "load-test"})
        await asyncio.sleep(poll_every)
    return {"job_id": job_id, "status": "TIMEOUT", "accept_ms": t_accept * 1000, "total_s": timeout, "error": None}


def p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20)[-1]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    tasks = [TASKS[i % len(TASKS)] for i in range(args.n)]
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(run_one(client, t, args.poll, args.timeout) for t in tasks))
        wall = time.perf_counter() - t0

    print(f"\n{'job_id':<34} {'status':<10} {'accept_ms':>10} {'total_s':>9}")
    for r in results:
        print(f"{r['job_id']:<34} {r['status']:<10} {r['accept_ms']:>10.1f} {r['total_s']:>9.1f}"
              + (f"  {r['error']}" if r["error"] else ""))
    accept = [r["accept_ms"] for r in results]
    total = [r["total_s"] for r in results if r["status"] == "DONE"]
    print(f"\nPeticiones: {len(results)} | DONE: {len(total)} | pared: {wall:.1f}s")
    print(f"Aceptación (POST /tasks) p95: {p95(accept):.1f} ms")
    if total:
        print(f"Latencia end-to-end p95: {p95(total):.1f} s | media: {statistics.mean(total):.1f} s")


if __name__ == "__main__":
    asyncio.run(main())
