"""Retriever agent — RAG over historical experiment logs."""

from __future__ import annotations

from typing import Any

from app.core.llm_client import LLMClient
from app.core.logging import get_logger
from app.rag.vector_store import VectorStore

logger = get_logger(__name__)


async def run_retriever(
    state: dict[str, Any],
    *,
    llm_client: LLMClient | None = None,
    vector_store: VectorStore | None = None,
) -> dict[str, Any]:
    """Retrieve relevant experiment logs for the given task description.

    Args:
        state: Orchestrator state containing task_description and llm_api_key.
        llm_client: Optional injected LLM client (for testing).
        vector_store: Optional injected vector store (for testing).

    Returns:
        Partial state update with retrieved_docs.
    """
    _ = llm_client  # reserved for query expansion via LLM
    store = vector_store or VectorStore()
    query = state.get("task_description", "")
    api_key = state.get("llm_api_key")

    docs = store.search(query, top_k=5, api_key=api_key)
    logger.info("Retriever found %d documents for query", len(docs))

    # Optional LLM summarization of retrieved context
    summary = ""
    client = llm_client or LLMClient()
    if api_key and docs:
        context = "\n".join(d.get("text", "") for d in docs[:3])
        try:
            summary = client.chat(
                [
                    {"role": "system", "content": "Summarize relevant experiment history briefly."},
                    {"role": "user", "content": f"Task: {query}\n\nLogs:\n{context}"},
                ],
                api_key=api_key,
                max_tokens=512,
            )
        except Exception as exc:
            logger.warning("Retriever LLM summarization skipped: %s", exc)

    return {
        "retrieved_docs": docs,
        "retrieval_summary": summary,
        "agents_completed": ["retriever"],
    }
