"""
app.py
======
FastAPI application for Cart Super Add-On recommendations.

Endpoints
---------
GET  /health              → liveness check + latency stats
POST /recommend           → get add-on recommendations for a cart
POST /recommend/sequential → sequential cart-update recommendations
GET  /metrics             → offline evaluation results
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.inference import AddonRecommendation, InferencePipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CartItem(BaseModel):
    item_id: str
    name: str
    category: str = ""
    price: float = 0.0
    qty: int = 1


class RecommendRequest(BaseModel):
    cart_items: list[CartItem]
    top_n: int = Field(default=5, ge=1, le=20)
    excluded_ids: list[str] = Field(default_factory=list)


class SequentialRequest(BaseModel):
    cart_history: list[list[CartItem]]
    top_n: int = Field(default=5, ge=1, le=20)


class RecommendResponse(BaseModel):
    recommendations: list[dict[str, Any]]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_size: int


# ---------------------------------------------------------------------------
# Application lifespan (load artefacts once)
# ---------------------------------------------------------------------------
pipeline = InferencePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts at startup, release at shutdown."""
    logger.info("Loading inference pipeline …")
    try:
        pipeline.load()
        logger.info("Pipeline ready ✓")
    except NotImplementedError:
        logger.warning("Pipeline not yet implemented — running in stub mode")
    yield
    logger.info("Shutting down …")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CartComplete — Super Add-On Recommender",
    version="0.1.0",
    description=(
        "LLM-augmented synthetic data + sentence-transformer embeddings "
        "+ LightGBM sequential ranker for Zomato Cart Super Add-On recs."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=pipeline.booster is not None,
        index_size=pipeline.faiss_index.ntotal if pipeline.faiss_index else 0,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    t0 = time.perf_counter()
    try:
        recs = pipeline.recommend(
            cart_items=[item.model_dump() for item in req.cart_items],
            top_n=req.top_n,
            excluded_ids=set(req.excluded_ids),
        )
    except NotImplementedError:
        raise HTTPException(503, "Pipeline not yet implemented")

    latency_ms = (time.perf_counter() - t0) * 1000
    return RecommendResponse(
        recommendations=[
            {
                "addon_id": r.addon_id,
                "addon_name": r.addon_name,
                "category": r.category,
                "price": r.price,
                "score": round(r.score, 4),
                "rank": r.rank,
            }
            for r in recs
        ],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/recommend/sequential", response_model=RecommendResponse)
async def recommend_sequential(req: SequentialRequest):
    t0 = time.perf_counter()
    try:
        recs = pipeline.recommend_sequential(
            cart_history=[[item.model_dump() for item in snap] for snap in req.cart_history],
            top_n=req.top_n,
        )
    except NotImplementedError:
        raise HTTPException(503, "Pipeline not yet implemented")

    latency_ms = (time.perf_counter() - t0) * 1000
    return RecommendResponse(
        recommendations=[
            {
                "addon_id": r.addon_id,
                "addon_name": r.addon_name,
                "category": r.category,
                "price": r.price,
                "score": round(r.score, 4),
                "rank": r.rank,
            }
            for r in recs
        ],
        latency_ms=round(latency_ms, 2),
    )
