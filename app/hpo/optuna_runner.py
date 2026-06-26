"""Optuna study runner for hyperparameter search."""

from __future__ import annotations

from typing import Any, Callable

import optuna

from app.config import get_settings
from app.core.logging import get_logger
from app.hpo.search_space import suggest_from_space

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_study(
    search_space: dict[str, dict[str, Any]],
    objective_fn: Callable[[dict[str, Any]], float],
    *,
    n_trials: int | None = None,
    study_name: str = "hpo",
) -> dict[str, Any]:
    """Run an Optuna study over a pruned search space.

    The objective function scores a hyperparameter dict (higher is better).
    This module does not execute training loops — it optimizes a surrogate
    objective supplied by the HPO agent.

    Returns:
        Dict with best_params, best_value, and trial history summary.
    """
    settings = get_settings()
    trials = n_trials or settings.hpo_n_trials

    def _objective(trial: optuna.Trial) -> float:
        params = suggest_from_space(trial, search_space)
        return objective_fn(params)

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(_objective, n_trials=trials, show_progress_bar=False)

    logger.info("Optuna study complete: best_value=%.4f", study.best_value)
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
