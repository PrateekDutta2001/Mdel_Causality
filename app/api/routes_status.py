"""Health and service status endpoints."""

from fastapi import APIRouter

from app import __version__

router = APIRouter(tags=["status"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


@router.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness probe.

    TODO: Ping Qdrant and MLflow before reporting ready.
    """
    return {"status": "ready"}
