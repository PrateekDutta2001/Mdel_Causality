"""Causal agent — estimate treatment effects over experiment hyperparameters."""

from __future__ import annotations

from typing import Any

from app.causal.effect_estimation import CausalEffect, estimate_causal_effects
from app.causal.graph_builder import get_causal_graph, load_experiment_data
from app.core.logging import get_logger

logger = get_logger(__name__)


async def run_causal(state: dict[str, Any]) -> dict[str, Any]:
    """Run causal effect estimation and produce a ranked parameter list.

    Returns:
        Partial state update with causal_rankings and causal_graph metadata.
    """
    graph = get_causal_graph()
    df = load_experiment_data()

    try:
        rankings: list[CausalEffect] = estimate_causal_effects(df, graph)
    except Exception as exc:
        logger.warning("DoWhy estimation failed, using correlation fallback: %s", exc)
        from app.causal.effect_estimation import fallback_correlation_effects

        rankings = fallback_correlation_effects(df)

    explanation = _build_explanation(rankings)
    logger.info("Causal agent ranked %d parameters", len(rankings))

    return {
        "causal_rankings": rankings,
        "causal_explanation": explanation,
        "agents_completed": ["causal"],
    }


def _build_explanation(rankings: list[CausalEffect]) -> str:
    lines = ["Causal effect ranking (parameter → effect_size, confidence):"]
    for item in rankings:
        lines.append(
            f"  - {item['parameter']}: effect={item['effect_size']:.4f}, "
            f"confidence={item['confidence']:.2f}"
        )
    return "\n".join(lines)
