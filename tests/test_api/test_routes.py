"""Tests for FastAPI selection endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_select_requires_api_key() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/select",
            json={"task_description": "classification"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_select_returns_run_id() -> None:
    with patch("app.api.routes_select._execute_run", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/select",
                json={
                    "task_description": "text classification",
                    "dataset_metadata": {},
                    "compute_budget": {},
                },
                headers={"X-LLM-API-Key": "test-key"},
            )
    assert response.status_code == 200
    body = response.json()
    assert "run_id" in body
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_get_run_not_found() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/runs/nonexistent-id")
    assert response.status_code == 404
