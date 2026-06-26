"""Evaluator agent — assess recommendation confidence and decide on HPO retry."""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.logging import get_logger

logger = get_logger(__name__)


async def run_evaluator(
    state: dict[str, Any],
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Evaluate the combined model + HPO recommendation.

    Sets evaluation.confidence; low confidence triggers a retry via orchestrator routing.

    Returns:
        Partial state update with evaluation and final_recommendation when confident.
    """
    settings = get_settings()
    client = llm_client or LLMClient()
    api_key = state.get("llm_api_key", "")

    selected = state.get("selected_model", {})
    hpo = state.get("hpo_result", {})
    causal = state.get("causal_explanation", "")

    confidence = _heuristic_confidence(selected, hpo, state.get("causal_rankings", []))

    rationale = ""
    if api_key:
        try:
            rationale = client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Evaluate an ML model + hyperparameter recommendation. "
                            "Provide a brief causal explanation and confidence 0-1."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Model: {selected}\nHPO: {hpo}\nCausal:\n{causal}\n"
                            f"Heuristic confidence: {confidence:.2f}"
                        ),
                    },
                ],
                api_key=api_key,
                max_tokens=768,
            )
        except Exception as exc:
            logger.warning("Evaluator LLM call failed: %s", exc)
            rationale = causal

    evaluation = {
        "confidence": confidence,
        "rationale": rationale or causal,
        "threshold": settings.evaluator_confidence_threshold,
    }

    final_recommendation = None
    if confidence >= settings.evaluator_confidence_threshold:
        final_recommendation = {
            "model": selected,
            "hyperparameters": hpo.get("best_params", {}),
            "expected_score": hpo.get("best_value"),
            "causal_report": causal,
            "evaluation": evaluation,
        }

    logger.info("Evaluator confidence=%.2f retry_count=%d", confidence, state.get("retry_count", 0))
    return {
        "evaluation": evaluation,
        "final_recommendation": final_recommendation,
        "agents_completed": ["evaluator"],
    }


def _heuristic_confidence(
    selected: dict[str, Any],
    hpo: dict[str, Any],
    causal_rankings: list[dict[str, Any]],
) -> float:
    score = 0.3
    if selected.get("model_id"):
        score += 0.25
    if hpo.get("best_params"):
        score += 0.25
    if causal_rankings:
        avg_conf = sum(r.get("confidence", 0) for r in causal_rankings) / len(causal_rankings)
        score += 0.2 * avg_conf
    return min(1.0, score)
