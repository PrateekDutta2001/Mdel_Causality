"""Tests for causal search-space pruning."""

from app.hpo.search_space import DEFAULT_SEARCH_SPACE, prune_search_space


def test_prune_drops_low_confidence_parameters() -> None:
    rankings = [
        {"parameter": "learning_rate", "effect_size": 0.12, "confidence": 0.85},
        {"parameter": "batch_size", "effect_size": 0.05, "confidence": 0.72},
        {"parameter": "num_epochs", "effect_size": 0.01, "confidence": 0.30},
        {"parameter": "optimizer", "effect_size": 0.02, "confidence": 0.25},
    ]

    pruned = prune_search_space(rankings, confidence_threshold=0.5)

    assert "learning_rate" in pruned
    assert "batch_size" in pruned
    assert "num_epochs" not in pruned
    assert "optimizer" not in pruned


def test_prune_shrinks_moderate_confidence_numeric_ranges() -> None:
    rankings = [
        {"parameter": "learning_rate", "effect_size": 0.1, "confidence": 0.55},
    ]
    base = {"learning_rate": dict(DEFAULT_SEARCH_SPACE["learning_rate"])}

    pruned = prune_search_space(rankings, confidence_threshold=0.5, base_space=base)
    original_span = base["learning_rate"]["high"] - base["learning_rate"]["low"]
    pruned_span = pruned["learning_rate"]["high"] - pruned["learning_rate"]["low"]

    assert pruned_span < original_span


def test_prune_keeps_all_when_confidence_high() -> None:
    rankings = [
        {"parameter": p, "effect_size": 0.1, "confidence": 0.9}
        for p in DEFAULT_SEARCH_SPACE
    ]
    pruned = prune_search_space(rankings, confidence_threshold=0.5)
    assert set(pruned.keys()) == set(DEFAULT_SEARCH_SPACE.keys())
