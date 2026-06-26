"""HPO agent — Optuna search over causally pruned hyperparameter space."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.hpo.optuna_runner import run_optuna_study
from app.hpo.search_space import prune_search_space

logger = get_logger(__name__)


def _surrogate_objective(params: dict[str, Any], causal_rankings: list[dict[str, Any]]) -> float:
    """Score hyperparameters using causal effect sizes (no training loop).

    Higher scores indicate hyperparameter values aligned with positive causal effects.
    """
    score = 0.0
    ranking_map = {r["parameter"]: r for r in causal_rankings}
    for name, value in params.items():
        effect = ranking_map.get(name)
        if effect is None:
            continue
        normalized = float(value) if isinstance(value, (int, float)) else 1.0
        score += effect["effect_size"] * normalized * effect["confidence"]
    return score


async def run_hpo(state: dict[str, Any]) -> dict[str, Any]:
    """Build a pruned search space and run Optuna optimization.

    Returns:
        Partial state update with hpo_result and pruned_search_space.
    """
    causal_rankings = state.get("causal_rankings", [])
    compute_budget = state.get("compute_budget", {})
    n_trials = compute_budget.get("max_hpo_trials")

    pruned_space = prune_search_space(causal_rankings)
    logger.info("HPO search space has %d parameters after causal pruning", len(pruned_space))

    def objective(params: dict[str, Any]) -> float:
        return _surrogate_objective(params, causal_rankings)

    study_result = run_optuna_study(pruned_space, objective, n_trials=n_trials)

    return {
        "pruned_search_space": pruned_space,
        "hpo_result": study_result,
        "agents_completed": ["hpo"],
    }
