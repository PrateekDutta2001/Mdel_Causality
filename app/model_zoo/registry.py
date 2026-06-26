"""Internal model registry abstraction for custom checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelEntry:
    """A registered model checkpoint."""

    model_id: str
    source: str  # "huggingface" | "local" | "custom"
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """In-memory registry for HF and custom model entries.

    TODO: Persist registry to disk or a database for production deployments.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}

    def register(self, entry: ModelEntry) -> None:
        self._entries[entry.model_id] = entry

    def get(self, model_id: str) -> ModelEntry | None:
        return self._entries.get(model_id)

    def list_by_task(self, task: str) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.task == task]

    def search_local(self, query: str) -> list[ModelEntry]:
        q = query.lower()
        return [
            e
            for e in self._entries.values()
            if q in e.model_id.lower() or q in e.task.lower()
        ]
