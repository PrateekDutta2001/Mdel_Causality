"""Embedding generation for RAG retrieval."""

from __future__ import annotations

from typing import Protocol

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for text embedding backends."""

    def embed(self, texts: list[str], *, api_key: str | None = None) -> list[list[float]]:
        """Return embedding vectors for each input text."""


class SentenceTransformerEmbeddings:
    """Local embeddings via sentence-transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str], *, api_key: str | None = None) -> list[list[float]]:
        self._load_model()
        assert self._model is not None
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


class OpenAIEmbeddings:
    """Remote embeddings via OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model

    def embed(self, texts: list[str], *, api_key: str | None = None) -> list[list[float]]:
        if not api_key:
            raise ValueError("OpenAI embeddings require api_key")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def get_embedding_provider() -> EmbeddingProvider:
    """Factory for the configured embedding backend."""
    settings = get_settings()
    if settings.embedding_backend == "openai":
        return OpenAIEmbeddings()
    return SentenceTransformerEmbeddings()
