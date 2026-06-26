"""Selection endpoints — kick off and poll orchestrator runs."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException

from app.agents.orchestrator import run_orchestrator
from app.api.schemas import SelectRequest, SelectResponse, RunStatusResponse
from app.config import get_settings
from app.core.logging import get_logger
from app.memory.mlflow_tracker import MLflowTracker
from app.memory.run_store import RunStatus, run_store

logger = get_logger(__name__)
router = APIRouter(tags=["selection"])


def _resolve_api_key(header_key: str | None) -> str:
    settings = get_settings()
    key = header_key or settings.llm_api_key
    if not key:
        raise HTTPException(
            status_code=401,
            detail="LLM API key required via X-LLM-API-Key header or LLM_API_KEY env var",
        )
    return key


async def _execute_run(run_id: str, request: SelectRequest, api_key: str) -> None:
    """Background task executing the LangGraph orchestrator."""
    await run_store.update_run(run_id, status=RunStatus.RUNNING)
    tracker = MLflowTracker()
    try:
        tracker.start_run(run_id, tags={"task": request.task_description[:128]})
        result = await run_orchestrator(
            run_id=run_id,
            task_description=request.task_description,
            dataset_metadata=request.dataset_metadata,
            compute_budget=request.compute_budget,
            llm_api_key=api_key,
        )
        for agent in result.get("agents_completed", []):
            await run_store.append_trace(run_id, agent)

        recommendation = result.get("final_recommendation") or {
            "model": result.get("selected_model"),
            "hyperparameters": (result.get("hpo_result") or {}).get("best_params"),
            "causal_report": result.get("causal_explanation"),
            "evaluation": result.get("evaluation"),
        }

        await run_store.update_run(
            run_id,
            status=RunStatus.COMPLETED,
            result={
                "recommendation": recommendation,
                "causal_report": result.get("causal_explanation"),
                "causal_rankings": result.get("causal_rankings"),
                "hpo_result": result.get("hpo_result"),
            },
        )
        tracker.log_params({"task": request.task_description})
        if recommendation.get("hyperparameters"):
            tracker.log_params(recommendation["hyperparameters"])
        tracker.end_run()
    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        await run_store.update_run(run_id, status=RunStatus.FAILED, error=str(exc))
        try:
            tracker.end_run(status="FAILED")
        except Exception:
            pass


@router.post("/select", response_model=SelectResponse)
async def select_model(
    request: SelectRequest,
    background_tasks: BackgroundTasks,
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
) -> SelectResponse:
    """Start an async model + hyperparameter selection pipeline."""
    api_key = _resolve_api_key(x_llm_api_key)
    run_id = await run_store.create_run(request.model_dump())
    background_tasks.add_task(_execute_run, run_id, request, api_key)
    return SelectResponse(run_id=run_id, status=RunStatus.PENDING.value)


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str) -> RunStatusResponse:
    """Poll run status and retrieve the final recommendation when complete."""
    run = await run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    result = run.get("result") or {}
    return RunStatusResponse(
        run_id=run_id,
        status=str(run.get("status", RunStatus.PENDING.value)),
        created_at=run.get("created_at"),
        updated_at=run.get("updated_at"),
        agent_trace=run.get("agent_trace", []),
        recommendation=result.get("recommendation"),
        causal_report=result.get("causal_report"),
        error=run.get("error"),
    )
