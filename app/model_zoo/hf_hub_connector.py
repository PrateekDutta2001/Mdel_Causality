"""HuggingFace Hub API client for pretrained checkpoint discovery."""

from __future__ import annotations

from typing import Any

from huggingface_hub import HfApi

from app.core.logging import get_logger

logger = get_logger(__name__)


class HFHubConnector:
    """Thin wrapper around the HuggingFace Hub API."""

    def __init__(self, token: str | None = None) -> None:
        self._api = HfApi(token=token)

    def search_models(
        self,
        query: str,
        *,
        task: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search Hub for models matching a natural-language query."""
        try:
            models = self._api.list_models(
                search=query,
                task=task,
                sort="downloads",
                direction=-1,
                limit=limit,
            )
        except Exception as exc:
            logger.warning("HF Hub search failed: %s", exc)
            return []

        return [
            {
                "model_id": m.modelId,
                "downloads": m.downloads,
                "tags": m.tags or [],
                "pipeline_tag": m.pipeline_tag,
            }
            for m in models
        ]

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Fetch metadata for a specific model."""
        info = self._api.model_info(model_id)
        return {
            "model_id": info.id,
            "tags": info.tags or [],
            "pipeline_tag": info.pipeline_tag,
            "siblings": [s.rfilename for s in (info.siblings or [])],
        }
