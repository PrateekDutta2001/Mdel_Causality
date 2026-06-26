"""API request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SelectRequest(BaseModel):
    """Payload for POST /select."""

    task_description: str = Field(..., description="Natural-language ML task description")
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
    compute_budget: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional limits, e.g. max_hpo_trials, max_gpu_hours",
    )


class SelectResponse(BaseModel):
    """Immediate response after kicking off an async selection run."""

    run_id: str
    status: str = "pending"


class RunStatusResponse(BaseModel):
    """Status and result for GET /runs/{run_id}."""

    run_id: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    agent_trace: list[str] = Field(default_factory=list)
    recommendation: dict[str, Any] | None = None
    causal_report: str | None = None
    error: str | None = None
