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

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.serving.orchestrator import ServingOrchestrator

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:
    Instrumentator = None

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
    user_id: str = "unknown"
    cart_items: list[CartItem]
    restaurant: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    top_n: int = Field(default=10, ge=1, le=30)
    mode: str = "balanced"


class SequentialRequest(BaseModel):
    cart_history: list[list[CartItem]]
    top_n: int = Field(default=10, ge=1, le=20)


class RecommendResponse(BaseModel):
    recommendations: list[dict[str, Any]]
    strategy: str
    latency_ms: float
    trace: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_size: int


# ---------------------------------------------------------------------------
# Application lifespan (load artefacts once)
# ---------------------------------------------------------------------------
orchestrator = ServingOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts at startup, release at shutdown."""
    logger.info("Initializing Serving Orchestrator …")
    # Pre-warm things if needed
    yield
    logger.info("Shutting down …")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CartComplete — Super Add-On Recommender",
    version="0.1.0",
    description=(
        "Production API Gateway for MealMind AI (Zomato Cart Super Add-On)."
    ),
    lifespan=lifespan,
)

if Instrumentator:
    Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=True,
        index_size=0,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, request: Request):
    t0 = time.perf_counter()
    
    # 1. Prepare inputs
    cart = [item.model_dump() for item in req.cart_items]
    rest = req.restaurant or {"restaurant_id": "R_default", "cuisine_type": "North Indian"}
    ctx = req.context or {"hour": 19, "meal_type": "dinner", "city": "Mumbai"}
    
    # 2. Invoke Orchestrator
    try:
        result = orchestrator.serve_request(
            cart_items=cart,
            restaurant=rest,
            context=ctx,
            top_k=req.top_n,
            session_id=str(request.headers.get("x-session-id", "")),
            mode=req.mode
        )
    except Exception as e:
        logger.exception("Orchestration failed")
        return RecommendResponse(
            recommendations=[],
            strategy="error",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    
    return RecommendResponse(
        recommendations=result.get("recommendations", []),
        strategy=result.get("strategy", "unknown"),
        latency_ms=round(latency_ms, 2),
        trace=result.get("trace", {})
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
