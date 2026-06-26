"""In-memory async run store for orchestrator job status."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStore:
    """Thread-safe in-memory store for orchestrator run state.

    TODO: Replace with Redis or PostgreSQL for multi-instance deployments.
    """

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_run(self, payload: dict[str, Any]) -> str:
        run_id = str(uuid4())
        async with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "status": RunStatus.PENDING,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "request": payload,
                "result": None,
                "error": None,
                "agent_trace": [],
            }
        return run_id

    async def update_run(self, run_id: str, **fields: Any) -> None:
        async with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Run {run_id} not found")
            self._runs[run_id].update(fields)
            self._runs[run_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    async def append_trace(self, run_id: str, agent: str) -> None:
        async with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["agent_trace"].append(agent)

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with self._lock:
            run = self._runs.get(run_id)
            return dict(run) if run else None


# Module-level singleton used by API routes.
run_store = RunStore()
