"""
inference.py
============
End-to-end inference pipeline for Cart Super Add-On recommendations.

Combines the three CSAO agents (MealContextAgent → RerankerAgent →
ColdStartAgent) with production infrastructure (caching, circuit
breakers, monitoring) via the ServingOrchestrator.

Flow (per request, target < 250 ms total)
------------------------------------------
1. Cache lookup (L1 in-process → L2 Redis)                (~1 ms)
2. Feature retrieval (cart analysis, user/restaurant)     (~30 ms)
3. Candidate generation (ColdStartAgent + menu KB)        (~50 ms)
4. LLM re-ranking (MealContextAgent → RerankerAgent)     (~150 ms)
5. Post-processing (business rules, diversity)             (~20 ms)
                                               TOTAL ≈ ~250 ms budget

Fallback chain (circuit-breaker managed):
  Full pipeline → Graph-only → CF-only → Cold Start → Popularity
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AddonRecommendation:
    """A single add-on recommendation returned by the pipeline."""
    addon_id: str
    addon_name: str
    category: str
    price: float
    score: float
    rank: int
    reasoning: str = ""


@dataclass
class InferencePipeline:
    """
    Holds pre-loaded artefacts and exposes ``recommend()`` /
    ``recommend_sequential()`` methods that run the full CSAO pipeline.

    Integrates:
      - MealContextAgent  (meal type, completion, cultural context)
      - RerankerAgent      (weighted scoring across 4 dimensions)
      - ColdStartAgent     (cuisine KB fallback for new users/restaurants)
      - ServingOrchestrator (caching, circuit breakers, monitoring)

    Designed to be instantiated **once** at server startup.
    """

    booster: Any = None
    faiss_index: Any = None
    addon_metadata: Any = None
    embedding_model: Any = None
    feature_columns: list[str] = field(default_factory=list)

    # Production components (lazy-loaded)
    _orchestrator: Any = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def load(self) -> "InferencePipeline":
        """
        Load all artefacts and initialise the serving orchestrator.

        In production this would load:
          - LightGBM booster from models/saved/
          - FAISS index from data/embeddings/
          - Sentence-transformer model (MiniLM-L6-v2)
          - Redis connections for feature store

        Currently loads the agent-based pipeline which needs no
        external artefacts — everything is built-in.
        """
        from .orchestrator import ServingOrchestrator
        from .production_config import ProductionConfig

        config = ProductionConfig()
        self._orchestrator = ServingOrchestrator(config)
        self._loaded = True
        logger.info("InferencePipeline loaded — orchestrator ready ✓")
        return self

    def recommend(
        self,
        cart_items: list[dict[str, Any]],
        top_n: int = 10,
        excluded_ids: set[str] | None = None,
        restaurant: dict[str, Any] | None = None,
        user: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AddonRecommendation]:
        """
        Given current cart items, return top-N add-on recommendations.

        Parameters
        ----------
        cart_items : list[dict]
            Each dict has keys: item_id, name, category, price, qty
        top_n : int
            Number of recommendations to return.
        excluded_ids : set[str]
            Add-on IDs already in the cart (to avoid duplicates).
        restaurant : dict, optional
            Restaurant details (cuisine_type, restaurant_type, etc.)
        user : dict, optional
            User preferences (dietary_preference, price_sensitivity)
        context : dict, optional
            Contextual info (hour, is_weekend, city)

        Returns
        -------
        list[AddonRecommendation]
        """
        if not self._loaded:
            self.load()

        # Infer restaurant info from cart if not provided
        if restaurant is None:
            restaurant = self._infer_restaurant(cart_items)

        # Run the full orchestrated pipeline
        result = self._orchestrator.serve_request(
            cart_items=cart_items,
            restaurant=restaurant,
            user=user,
            context=context,
            top_k=top_n,
        )

        # Convert to AddonRecommendation objects
        recs = result.get("recommendations", [])
        excluded = excluded_ids or set()

        output: list[AddonRecommendation] = []
        rank = 0
        for item in recs:
            item_id = item.get("item_id", item.get("name", "")[:8])
            if item_id in excluded:
                continue
            rank += 1
            output.append(AddonRecommendation(
                addon_id=item_id,
                addon_name=item.get("name", ""),
                category=item.get("category", ""),
                price=item.get("price", 0.0),
                score=round(item.get("final_score", item.get("confidence", 0.5)), 4),
                rank=rank,
                reasoning=item.get("reasoning", item.get("why_this_item_now", "")),
            ))
            if rank >= top_n:
                break

        return output

    def recommend_sequential(
        self,
        cart_history: list[list[dict[str, Any]]],
        top_n: int = 10,
        restaurant: dict[str, Any] | None = None,
        user: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AddonRecommendation]:
        """
        Handle sequential cart updates.

        ``cart_history`` is an ordered list of cart snapshots.
        The latest snapshot is used for ranking; earlier snapshots
        contribute a momentum / recency signal.

        On each cart mutation, we re-run the full pipeline with
        previously-added items excluded from candidates.
        """
        if not cart_history:
            return []

        # Use the latest snapshot as the active cart
        current_cart = cart_history[-1]

        # Build exclusion set from all items across all snapshots
        all_ids = set()
        for snapshot in cart_history:
            for item in snapshot:
                all_ids.add(item.get("item_id", item.get("name", "")))

        return self.recommend(
            cart_items=current_cart,
            top_n=top_n,
            excluded_ids=all_ids,
            restaurant=restaurant,
            user=user,
            context=context,
        )

    def health(self) -> dict:
        """Return system health snapshot."""
        if not self._loaded:
            return {"status": "not_loaded", "model_loaded": False, "index_size": 0}

        orch_health = self._orchestrator.health_check()
        return {
            "status": orch_health.get("status", "ok"),
            "model_loaded": self._loaded,
            "index_size": 0,  # FAISS not used in agent pipeline
            "active_strategy": orch_health.get("active_strategy", "unknown"),
            "circuit_breakers": orch_health.get("circuit_breakers", {}),
            "cache": orch_health.get("cache", {}),
        }

    @staticmethod
    def _infer_restaurant(cart_items: list[dict]) -> dict:
        """Infer restaurant type from cart contents when not provided."""
        categories = [i.get("category", "").lower() for i in cart_items]
        names = " ".join(i.get("name", "").lower() for i in cart_items)

        cuisine = "North Indian"  # default
        if any(w in names for w in ["biryani", "kebab", "salan"]):
            cuisine = "Biryani Specialist"
        elif any(w in names for w in ["dosa", "idli", "sambhar", "vada"]):
            cuisine = "South Indian"
        elif any(w in names for w in ["noodles", "manchurian", "spring roll"]):
            cuisine = "Chinese"
        elif any(w in names for w in ["pizza", "pasta", "burger"]):
            cuisine = "Continental"
        elif any(w in names for w in ["fries", "nuggets", "mcflurry"]):
            cuisine = "Fast Food"

        avg_price = sum(i.get("price", 0) for i in cart_items) / max(len(cart_items), 1)
        price_tier = "budget" if avg_price < 150 else ("premium" if avg_price > 400 else "mid")

        return {
            "cuisine_type": cuisine,
            "price_tier": price_tier,
            "restaurant_type": "local",
        }
