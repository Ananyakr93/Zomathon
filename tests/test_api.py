"""
test_api.py
===========
Integration tests for the FastAPI serving layer.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from src.serving.app import app


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_recommend_stub_returns_503():
    """Before pipeline is implemented, /recommend should return 503."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/recommend", json={
            "cart_items": [
                {"item_id": "1", "name": "Butter Chicken", "category": "main", "price": 349, "qty": 1}
            ],
            "top_n": 3,
        })
        assert resp.status_code == 503
