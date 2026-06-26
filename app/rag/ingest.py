"""Ingest experiment logs and documentation into the vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_settings
from app.core.logging import get_logger
from app.rag.vector_store import VectorStore

logger = get_logger(__name__)

LOG_COLUMNS = [
    "architecture",
    "dataset_size",
    "learning_rate",
    "batch_size",
    "optimizer",
    "num_epochs",
    "hardware_type",
    "final_metric",
]


def _row_to_text(row: pd.Series) -> str:
    """Serialize an experiment log row into a retrieval-friendly text chunk."""
    parts = [f"{col}={row[col]}" for col in LOG_COLUMNS if col in row.index]
    return " | ".join(parts)


def load_experiment_logs(logs_dir: str | None = None) -> pd.DataFrame:
    """Load all CSV/Parquet files under the experiment logs directory."""
    settings = get_settings()
    base = Path(logs_dir or settings.experiment_logs_dir)
    if not base.exists():
        logger.warning("Experiment logs directory not found: %s", base)
        return pd.DataFrame(columns=LOG_COLUMNS)

    frames: list[pd.DataFrame] = []
    for path in sorted(base.glob("**/*")):
        if path.suffix == ".csv":
            frames.append(pd.read_csv(path))
        elif path.suffix == ".parquet":
            frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_documents_from_logs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert tabular experiment logs into vector-store documents."""
    documents: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        documents.append(
            {
                "text": _row_to_text(row),
                "source": "experiment_log",
                **{col: row.get(col) for col in LOG_COLUMNS if col in row.index},
            }
        )
    return documents


def ingest_experiment_logs(
    *,
    logs_dir: str | None = None,
    api_key: str | None = None,
    vector_store: VectorStore | None = None,
) -> int:
    """Load local experiment logs and upsert them into Qdrant.

    Returns:
        Number of documents ingested.
    """
    df = load_experiment_logs(logs_dir)
    if df.empty:
        logger.info("No experiment logs to ingest")
        return 0

    docs = build_documents_from_logs(df)
    store = vector_store or VectorStore()
    count = store.upsert_documents(docs, api_key=api_key)
    logger.info("Ingested %d experiment log documents", count)
    return count
