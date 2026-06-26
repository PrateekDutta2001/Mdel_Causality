"""LangGraph orchestrator wiring retriever, causal, model selector, HPO, and evaluator."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agents.causal_agent import run_causal
from app.agents.evaluator_agent import run_evaluator
from app.agents.hpo_agent import run_hpo
from app.agents.model_selector_agent import run_model_selector
from app.agents.retriever_agent import run_retriever
from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.logging import get_logger

logger = get_logger(__name__)


class OrchestratorState(TypedDict, total=False):
    """Shared state passed between LangGraph agent nodes."""

    run_id: str
    task_description: str
    dataset_metadata: dict[str, Any]
    compute_budget: dict[str, Any]
    llm_api_key: str

    retrieved_docs: list[dict[str, Any]]
    retrieval_summary: str
    causal_rankings: list[dict[str, Any]]
    causal_explanation: str
    selected_model: dict[str, Any]
    pruned_search_space: dict[str, Any]
    hpo_result: dict[str, Any]
    evaluation: dict[str, Any]
    final_recommendation: dict[str, Any] | None

    retry_count: int
    agents_completed: Annotated[list[str], operator.add]
    parallel_done: Annotated[list[str], operator.add]


def _sync_barrier(state: OrchestratorState) -> dict[str, Any]:
    """No-op join node; LangGraph waits for all inbound edges before executing."""
    return {}


def _route_after_evaluator(
    state: OrchestratorState,
) -> Literal["hpo", "__end__"]:
    """Route back to HPO on low confidence, up to max retries."""
    settings = get_settings()
    evaluation = state.get("evaluation") or {}
    confidence = evaluation.get("confidence", 1.0)
    retries = state.get("retry_count", 0)

    if confidence < settings.evaluator_confidence_threshold and retries < settings.evaluator_max_retries:
        logger.info("Low confidence (%.2f); routing to HPO retry %d", confidence, retries + 1)
        return "hpo"
    return "__end__"


def _increment_retry(state: OrchestratorState) -> dict[str, Any]:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_orchestrator_graph(
    *,
    llm_client: LLMClient | None = None,
) -> Any:
    """Construct and compile the LangGraph StateGraph.

    Flow:
        START → retriever ∥ causal (parallel)
              → sync_barrier → model_selector → hpo → evaluator
              → (retry hpo if low confidence, max 2 retries) → END

    Args:
        llm_client: Optional injected LLM client for testing.

    Returns:
        Compiled LangGraph runnable.
    """
    client = llm_client or LLMClient()

    async def retriever_node(state: OrchestratorState) -> dict[str, Any]:
        result = await run_retriever(state, llm_client=client)
        return {**result, "parallel_done": ["retriever"]}

    async def causal_node(state: OrchestratorState) -> dict[str, Any]:
        result = await run_causal(state)
        return {**result, "parallel_done": ["causal"]}

    async def model_selector_node(state: OrchestratorState) -> dict[str, Any]:
        return await run_model_selector(state, llm_client=client)

    async def hpo_node(state: OrchestratorState) -> dict[str, Any]:
        return await run_hpo(state)

    async def evaluator_node(state: OrchestratorState) -> dict[str, Any]:
        return await run_evaluator(state, llm_client=client)

    graph = StateGraph(OrchestratorState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("causal", causal_node)
    graph.add_node("sync_barrier", _sync_barrier)
    graph.add_node("model_selector", model_selector_node)
    graph.add_node("hpo", hpo_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("increment_retry", _increment_retry)

    # Parallel fan-out from START
    graph.add_edge(START, "retriever")
    graph.add_edge(START, "causal")

    # Fan-in before model selection
    graph.add_edge("retriever", "sync_barrier")
    graph.add_edge("causal", "sync_barrier")
    graph.add_edge("sync_barrier", "model_selector")

    graph.add_edge("model_selector", "hpo")
    graph.add_edge("hpo", "evaluator")

    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {"hpo": "increment_retry", "__end__": END},
    )
    graph.add_edge("increment_retry", "hpo")

    return graph.compile()


async def run_orchestrator(
    *,
    run_id: str,
    task_description: str,
    dataset_metadata: dict[str, Any],
    compute_budget: dict[str, Any],
    llm_api_key: str,
    llm_client: LLMClient | None = None,
) -> OrchestratorState:
    """Execute the full orchestrator pipeline for a selection request."""
    app = build_orchestrator_graph(llm_client=llm_client)
    initial: OrchestratorState = {
        "run_id": run_id,
        "task_description": task_description,
        "dataset_metadata": dataset_metadata,
        "compute_budget": compute_budget,
        "llm_api_key": llm_api_key,
        "retry_count": 0,
        "agents_completed": [],
        "parallel_done": [],
    }
    result: OrchestratorState = await app.ainvoke(initial)
    logger.info(
        "Orchestrator complete run_id=%s agents=%s",
        run_id,
        result.get("agents_completed"),
    )
    return result
