"""Qdrant vector store wrapper for experiment log retrieval."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings import get_embedding_provider

logger = get_logger(__name__)


class VectorStore:
    """Thin abstraction over Qdrant for upsert and similarity search."""

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._url = url or settings.qdrant_url
        self._collection = collection_name or settings.qdrant_collection
        self._client = QdrantClient(url=self._url)
        self._embedder = get_embedding_provider()
        self._vector_size: int | None = None

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it does not exist."""
        self._vector_size = vector_size
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            logger.info("Creating Qdrant collection %s (dim=%d)", self._collection, vector_size)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )

    def upsert_documents(
        self,
        documents: list[dict[str, Any]],
        *,
        api_key: str | None = None,
    ) -> int:
        """Embed and upsert document payloads into Qdrant."""
        if not documents:
            return 0

        texts = [doc.get("text", "") for doc in documents]
        vectors = self._embedder.embed(texts, api_key=api_key)
        self.ensure_collection(len(vectors[0]))

        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=doc,
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        api_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k similar documents for a natural-language query."""
        vector = self._embedder.embed([query], api_key=api_key)[0]
        if self._vector_size is None:
            self._vector_size = len(vector)

        try:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
            )
        except Exception as exc:
            logger.warning("Qdrant search failed (%s); returning empty results", exc)
            return []

        return [
            {
                "score": hit.score,
                **hit.payload,
            }
            for hit in results
            if hit.payload is not None
        ]
