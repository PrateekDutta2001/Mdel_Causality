"""MLflow experiment tracking integration."""

from __future__ import annotations

from typing import Any

import mlflow

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """Log orchestrator runs and recommendations to MLflow."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self._experiment_name = experiment_name or settings.mlflow_experiment_name
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)

    def start_run(self, run_id: str, tags: dict[str, str] | None = None) -> str:
        """Start an MLflow run tagged with the internal run_id."""
        run = mlflow.start_run(run_name=run_id, tags=tags or {})
        mlflow.set_tag("internal_run_id", run_id)
        return run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {k: str(v) for k, v in params.items()}
        mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_artifact_text(self, name: str, text: str) -> None:
        mlflow.log_text(text, name)

    def end_run(self, status: str = "FINISHED") -> None:
        mlflow.end_run(status=status)
