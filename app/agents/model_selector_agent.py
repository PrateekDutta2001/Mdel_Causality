"""Model selector agent — recommend a pretrained checkpoint."""

from __future__ import annotations

import json
from typing import Any

from app.core.llm_client import LLMClient
from app.core.logging import get_logger
from app.model_zoo.hf_hub_connector import HFHubConnector
from app.model_zoo.registry import ModelEntry, ModelRegistry

logger = get_logger(__name__)


async def run_model_selector(
    state: dict[str, Any],
    *,
    llm_client: LLMClient | None = None,
    hf_connector: HFHubConnector | None = None,
    registry: ModelRegistry | None = None,
) -> dict[str, Any]:
    """Select a pretrained model given task description and retrieved context.

    Returns:
        Partial state update with selected_model recommendation.
    """
    client = llm_client or LLMClient()
    connector = hf_connector or HFHubConnector()
    reg = registry or ModelRegistry()
    api_key = state.get("llm_api_key", "")

    task = state.get("task_description", "")
    dataset_meta = state.get("dataset_metadata", {})
    retrieved = state.get("retrieved_docs", [])
    causal_summary = state.get("causal_explanation", "")

    hub_candidates = connector.search_models(task, limit=5)
    for candidate in hub_candidates:
        reg.register(
            ModelEntry(
                model_id=candidate["model_id"],
                source="huggingface",
                task=task,
                metadata=candidate,
            )
        )

    prompt = _build_selection_prompt(task, dataset_meta, retrieved, causal_summary, hub_candidates)
    recommendation: dict[str, Any]

    if api_key:
        try:
            raw = client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an ML architect. Recommend one pretrained model "
                            "checkpoint and explain why. Respond in JSON with keys: "
                            "model_id, rationale, expected_metric_range."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                max_tokens=1024,
            )
            recommendation = _parse_recommendation(raw, hub_candidates)
        except Exception as exc:
            logger.warning("Model selector LLM call failed: %s", exc)
            recommendation = _fallback_selection(hub_candidates)
    else:
        recommendation = _fallback_selection(hub_candidates)

    logger.info("Model selector chose: %s", recommendation.get("model_id"))
    return {
        "selected_model": recommendation,
        "agents_completed": ["model_selector"],
    }


def _build_selection_prompt(
    task: str,
    dataset_meta: dict[str, Any],
    retrieved: list[dict[str, Any]],
    causal_summary: str,
    candidates: list[dict[str, Any]],
) -> str:
    return (
        f"Task: {task}\n"
        f"Dataset metadata: {json.dumps(dataset_meta)}\n"
        f"Retrieved experiments: {json.dumps(retrieved[:3])}\n"
        f"Causal analysis:\n{causal_summary}\n"
        f"HF Hub candidates: {json.dumps(candidates)}\n"
    )


def _parse_recommendation(raw: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except json.JSONDecodeError:
        pass
    return _fallback_selection(candidates)


def _fallback_selection(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if candidates:
        top = candidates[0]
        return {
            "model_id": top["model_id"],
            "rationale": "Selected by Hub download ranking (LLM unavailable).",
            "expected_metric_range": "unknown",
        }
    return {
        "model_id": "distilbert-base-uncased",
        "rationale": "Default fallback checkpoint.",
        "expected_metric_range": "unknown",
    }
