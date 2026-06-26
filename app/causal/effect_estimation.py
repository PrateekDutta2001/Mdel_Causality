"""Average treatment effect estimation with DoWhy."""

from __future__ import annotations

from typing import Any, TypedDict

import networkx as nx
import numpy as np
import pandas as pd

from app.causal.confounder_checks import get_confounders
from app.causal.graph_builder import OUTCOME, TREATMENT_PARAMS, get_causal_graph, load_experiment_data
from app.core.logging import get_logger

logger = get_logger(__name__)


class CausalEffect(TypedDict):
    """Ranked causal effect for a hyperparameter."""

    parameter: str
    effect_size: float
    confidence: float


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals and coerce numerics for causal estimation."""
    work = df.copy()
    for col in work.select_dtypes(include=["object", "string"]).columns:
        work[col] = work[col].astype("category").cat.codes
    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.dropna()


def _estimate_single_effect(
    df: pd.DataFrame,
    graph: nx.DiGraph,
    treatment: str,
) -> CausalEffect:
    """Estimate ATE for one treatment variable using DoWhy."""
    from dowhy import CausalModel

    confounders = [c for c in get_confounders(graph, treatment) if c in df.columns]
    if treatment not in df.columns or OUTCOME not in df.columns:
        return {"parameter": treatment, "effect_size": 0.0, "confidence": 0.0}

    common = confounders + [treatment, OUTCOME]
    subset = df[common].dropna()
    if len(subset) < 5:
        logger.warning("Insufficient data for treatment=%s (n=%d)", treatment, len(subset))
        return {"parameter": treatment, "effect_size": 0.0, "confidence": 0.1}

    gml = _subgraph_gml(graph, common)
    model = CausalModel(data=subset, treatment=treatment, outcome=OUTCOME, graph=gml)
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )

    effect_size = float(getattr(estimate, "value", 0.0) or 0.0)
    p_value = _extract_p_value(estimate)
    confidence = float(max(0.0, min(1.0, 1.0 - p_value)))

    return {
        "parameter": treatment,
        "effect_size": effect_size,
        "confidence": confidence,
    }


def _subgraph_gml(graph: nx.DiGraph, nodes: list[str]) -> str:
    """Extract an induced subgraph GML string."""
    import io

    sub = graph.subgraph(nodes).copy()
    buffer = io.StringIO()
    nx.write_gml(sub, buffer)
    return buffer.getvalue()


def _extract_p_value(estimate: Any) -> float:
    """Best-effort p-value extraction from a DoWhy estimate object."""
    for attr in ("p_value", "pvalue"):
        if hasattr(estimate, attr):
            val = getattr(estimate, attr)
            if val is not None:
                return float(val)
    test = getattr(estimate, "test_stat_significance", None)
    if callable(test):
        try:
            result = test()
            if isinstance(result, dict) and "p_value" in result:
                return float(result["p_value"])
        except Exception:
            pass
    return 0.5


def estimate_causal_effects(
    df: pd.DataFrame | None = None,
    graph: nx.DiGraph | None = None,
) -> list[CausalEffect]:
    """Estimate and rank average treatment effects for each hyperparameter.

    Returns:
        List sorted by absolute effect size descending.
    """
    graph = graph or get_causal_graph()
    raw = df if df is not None else load_experiment_data()
    if raw.empty:
        logger.warning("No experiment data; returning zero-effect placeholders")
        return [
            {"parameter": p, "effect_size": 0.0, "confidence": 0.0}
            for p in TREATMENT_PARAMS
        ]

    prepared = _prepare_dataframe(raw)
    effects: list[CausalEffect] = []
    for param in TREATMENT_PARAMS:
        try:
            effects.append(_estimate_single_effect(prepared, graph, param))
        except Exception as exc:
            logger.warning("DoWhy estimation failed for %s: %s", param, exc)
            effects.append({"parameter": param, "effect_size": 0.0, "confidence": 0.0})

    effects.sort(key=lambda e: abs(e["effect_size"]), reverse=True)
    return effects


def fallback_correlation_effects(df: pd.DataFrame) -> list[CausalEffect]:
    """Lightweight fallback when DoWhy is unavailable (used in tests)."""
    if df.empty or OUTCOME not in df.columns:
        return [{"parameter": p, "effect_size": 0.0, "confidence": 0.0} for p in TREATMENT_PARAMS]

    work = _prepare_dataframe(df)
    effects: list[CausalEffect] = []
    for param in TREATMENT_PARAMS:
        if param not in work.columns:
            effects.append({"parameter": param, "effect_size": 0.0, "confidence": 0.0})
            continue
        corr = float(work[param].corr(work[OUTCOME]))
        if np.isnan(corr):
            corr = 0.0
        effects.append(
            {
                "parameter": param,
                "effect_size": corr,
                "confidence": min(1.0, abs(corr) + 0.3),
            }
        )
    effects.sort(key=lambda e: abs(e["effect_size"]), reverse=True)
    return effects
