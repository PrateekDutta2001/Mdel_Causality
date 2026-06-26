"""Build a causal DAG over experiment hyperparameters and outcomes."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Plausible causal edges among experiment variables.
DEFAULT_EDGES: list[tuple[str, str]] = [
    ("hardware_type", "batch_size"),
    ("hardware_type", "num_epochs"),
    ("dataset_size", "learning_rate"),
    ("dataset_size", "batch_size"),
    ("architecture", "learning_rate"),
    ("architecture", "final_metric"),
    ("learning_rate", "final_metric"),
    ("batch_size", "final_metric"),
    ("optimizer", "final_metric"),
    ("num_epochs", "final_metric"),
    ("hardware_type", "final_metric"),
    ("dataset_size", "final_metric"),
]

TREATMENT_PARAMS = [
    "learning_rate",
    "batch_size",
    "optimizer",
    "num_epochs",
    "architecture",
]

OUTCOME = "final_metric"


def load_experiment_data(logs_dir: str | None = None) -> pd.DataFrame:
    """Load experiment logs from disk."""
    settings = get_settings()
    base = Path(logs_dir or settings.experiment_logs_dir)
    frames: list[pd.DataFrame] = []
    if base.exists():
        for path in sorted(base.glob("**/*.csv")):
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_causal_dag(edges: list[tuple[str, str]] | None = None) -> nx.DiGraph:
    """Construct a directed acyclic graph encoding plausible causal relationships."""
    graph = nx.DiGraph()
    for src, dst in edges or DEFAULT_EDGES:
        graph.add_edge(src, dst)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Configured causal edges contain a cycle")
    return graph


def get_causal_graph(logs_dir: str | None = None) -> nx.DiGraph:
    """Return the causal DAG, optionally validating nodes against available log columns."""
    graph = build_causal_dag()
    df = load_experiment_data(logs_dir)
    if not df.empty:
        missing = [n for n in graph.nodes if n not in df.columns]
        if missing:
            logger.debug("DAG nodes absent from data (kept for structure): %s", missing)
    return graph


def dag_to_gml(graph: nx.DiGraph) -> str:
    """Serialize the DAG to GML for DoWhy consumption."""
    import io

    buffer = io.StringIO()
    nx.write_gml(graph, buffer)
    return buffer.getvalue()
