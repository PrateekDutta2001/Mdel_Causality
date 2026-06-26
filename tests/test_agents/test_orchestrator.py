"""Tests for the LangGraph orchestrator pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.agents.orchestrator import build_orchestrator_graph, run_orchestrator
from app.core.llm_client import LLMClient


class MockLLMClient(LLMClient):
    """Mock LLM that returns deterministic JSON/text without network calls."""

    def __init__(self) -> None:
        super().__init__(provider="openai")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        api_key: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        _ = messages, model, temperature, max_tokens, kwargs
        assert api_key == "test-key"
        return (
            '{"model_id": "distilbert-base-uncased", '
            '"rationale": "Good for classification", '
            '"expected_metric_range": "0.85-0.90"}'
        )


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.search.return_value = [
        {
            "text": "architecture=bert-base | learning_rate=0.0001 | final_metric=0.91",
            "score": 0.95,
            "architecture": "bert-base",
            "final_metric": 0.91,
        }
    ]
    return store


@pytest.fixture
def mock_hf_connector() -> MagicMock:
    connector = MagicMock()
    connector.search_models.return_value = [
        {"model_id": "distilbert-base-uncased", "downloads": 1000, "tags": [], "pipeline_tag": "text-classification"}
    ]
    return connector


@pytest.mark.asyncio
async def test_orchestrator_routes_all_five_agents(
    mock_vector_store: MagicMock,
    mock_hf_connector: MagicMock,
) -> None:
    """Verify the graph executes retriever, causal, model_selector, hpo, and evaluator."""
    llm = MockLLMClient()
    mock_rankings = [
        {"parameter": "learning_rate", "effect_size": 0.12, "confidence": 0.85},
        {"parameter": "batch_size", "effect_size": 0.05, "confidence": 0.72},
        {"parameter": "num_epochs", "effect_size": 0.03, "confidence": 0.65},
        {"parameter": "optimizer", "effect_size": 0.02, "confidence": 0.60},
    ]

    with (
        patch("app.agents.retriever_agent.VectorStore", return_value=mock_vector_store),
        patch("app.agents.model_selector_agent.HFHubConnector", return_value=mock_hf_connector),
        patch("app.agents.causal_agent.estimate_causal_effects", return_value=mock_rankings),
        patch("app.agents.evaluator_agent.get_settings") as mock_settings,
    ):
        settings = MagicMock()
        settings.evaluator_confidence_threshold = 0.5
        settings.evaluator_max_retries = 2
        mock_settings.return_value = settings

        result = await run_orchestrator(
            run_id="test-run",
            task_description="text classification",
            dataset_metadata={"num_samples": 1000},
            compute_budget={"max_hpo_trials": 3},
            llm_api_key="test-key",
            llm_client=llm,
        )

    completed = result.get("agents_completed", [])
    for agent in ["retriever", "causal", "model_selector", "hpo", "evaluator"]:
        assert agent in completed, f"Expected {agent} in completed agents, got {completed}"

    assert result.get("selected_model", {}).get("model_id")
    assert result.get("hpo_result", {}).get("best_params")
    assert result.get("causal_rankings")


@pytest.mark.asyncio
async def test_orchestrator_graph_compiles() -> None:
    graph = build_orchestrator_graph(llm_client=MockLLMClient())
    assert graph is not None
