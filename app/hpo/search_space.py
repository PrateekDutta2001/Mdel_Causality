"""Build Optuna search spaces pruned by causal effect rankings."""

from __future__ import annotations

from typing import Any

import optuna

from app.causal.effect_estimation import CausalEffect
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Full default ranges before causal pruning.
DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
    "batch_size": {"type": "int", "low": 8, "high": 256, "step": 8},
    "num_epochs": {"type": "int", "low": 1, "high": 100},
    "optimizer": {
        "type": "categorical",
        "choices": ["adam", "adamw", "sgd", "rmsprop"],
    },
}


def _is_low_confidence(effect: CausalEffect, threshold: float) -> bool:
    return effect["confidence"] < threshold


def prune_search_space(
    causal_rankings: list[CausalEffect],
    *,
    confidence_threshold: float | None = None,
    base_space: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Drop or shrink parameters whose causal confidence is below threshold.

    Low-confidence parameters are removed from the search space entirely.
    Medium-confidence parameters have their ranges narrowed toward observed
    effect direction (shrink by 50%).

    Args:
        causal_rankings: Ranked list from ``estimate_causal_effects``.
        confidence_threshold: Minimum confidence to keep full range.
        base_space: Optional override of default hyperparameter ranges.

    Returns:
        Pruned search space specification.
    """
    settings = get_settings()
    threshold = confidence_threshold if confidence_threshold is not None else settings.causal_confidence_threshold
    space = dict(base_space or DEFAULT_SEARCH_SPACE)
    ranking_map = {e["parameter"]: e for e in causal_rankings}

    pruned: dict[str, dict[str, Any]] = {}
    for param, spec in space.items():
        effect = ranking_map.get(param)
        if effect is None:
            pruned[param] = spec
            continue

        if _is_low_confidence(effect, threshold):
            logger.info(
                "Dropping parameter %s from HPO (confidence=%.2f < %.2f)",
                param,
                effect["confidence"],
                threshold,
            )
            continue

        if effect["confidence"] < threshold + 0.2 and spec["type"] in ("float", "int"):
            shrunk = dict(spec)
            if spec["type"] == "float":
                mid = (spec["low"] + spec["high"]) / 2
                span = (spec["high"] - spec["low"]) * 0.25
                shrunk["low"] = max(spec["low"], mid - span)
                shrunk["high"] = min(spec["high"], mid + span)
            else:
                mid = (spec["low"] + spec["high"]) // 2
                span = max(1, (spec["high"] - spec["low"]) // 4)
                shrunk["low"] = max(spec["low"], mid - span)
                shrunk["high"] = min(spec["high"], mid + span)
            pruned[param] = shrunk
            logger.info("Shrunk search range for %s due to moderate causal confidence", param)
        else:
            pruned[param] = spec

    return pruned


def suggest_from_space(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Suggest hyperparameters from a pruned space definition."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params
