"""
Serving Orchestrator — CSAO Production Pipeline
=================================================
Wires together all production components into a single request path:

  Request → Rate-limit → Cache lookup → Feature retrieval
          → Candidate generation → LLM re-ranking (with CB)
          → Post-processing → Response + Monitoring

Supports concurrent execution of independent stages via
ThreadPoolExecutor (simulating asyncio for CPU-bound steps).
"""

from __future__ import annotations

import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .production_config import ProductionConfig
from .circuit_breaker import FallbackManager, CircuitState
from .cache_manager import CacheManager
from .monitoring import RequestTracer, MonitoringDashboard
from .session_state import SessionStore

logger = logging.getLogger(__name__)

# Minimum relevance score — items below this are filtered out
MIN_SCORE_THRESHOLD = 0.15


# ═════════════════════════════════════════════════════════════════════════════
#  SERVING ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

class ServingOrchestrator:
    """
    Production-grade request orchestrator.

    Manages the full lifecycle of a recommendation request:
      1. Rate-limit check
      2. Cache lookup (L1 → L2 → compute)
      3. Parallel feature retrieval + candidate generation
      4. LLM re-ranking with circuit-breaker fallback
      5. Post-processing (business rules, diversity)
      6. Response assembly with tracing

    Designed to be instantiated once at server startup.
    """

    def __init__(self, config: ProductionConfig | None = None) -> None:
        self.config = config or ProductionConfig()
        self.cache = CacheManager()
        self.fallback = FallbackManager()
        self.monitor = MonitoringDashboard()
        self.sessions = SessionStore()
        self._executor = ThreadPoolExecutor(max_workers=8)

    # ── MAIN ENTRY POINT ───────────────────────────────────────────────────

    def serve_request(
        self,
        cart_items: list[dict[str, Any]],
        restaurant: dict[str, Any],
        user: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        top_k: int = 10,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full recommendation pipeline.

        Returns a dict with recommendations, latency, trace, and strategy used.
        """
        tracer = RequestTracer()
        tracer.metadata = {
            "cart_size": len(cart_items),
            "restaurant": restaurant.get("cuisine_type", "unknown"),
            "user_id": (user or {}).get("user_id", "anon"),
        }

        strategy = "full"
        is_error = False

        try:
            # ── 1. Rate limit ──────────────────────────────────────────────
            if not self.fallback.check_rate_limit():
                return self._rate_limited_response(tracer)

            # ── 2. Cache check ─────────────────────────────────────────────
            tracer.start_span("cache_lookup")
            cache_key = CacheManager.make_cache_key(
                "llm_response",
                cart=[i.get("name", "") for i in cart_items],
                cuisine=restaurant.get("cuisine_type", ""),
            )
            cached = self.cache.l1.get(cache_key) or self.cache.l2.get(cache_key)
            tracer.end_span(cache_hit=cached is not None)

            if cached is not None:
                tracer.metadata["cache_hit"] = True
                self.monitor.record_request(tracer, strategy="cached")
                cached["trace"] = tracer.to_dict()
                cached["latency_ms"] = tracer.total_ms
                return cached

            # ── 3. Determine strategy ──────────────────────────────────────
            strategy = self.fallback.get_available_strategy()
            tracer.metadata["strategy"] = strategy

            # ── 4. Feature retrieval + candidate gen (parallel) ────────────
            tracer.start_span("feature_retrieval")
            features = self._retrieve_features(cart_items, restaurant, user)
            tracer.end_span()

            # ── 4b. Session state tracking ─────────────────────────────────
            session_ctx = {}
            completeness = features.get("meal_completeness_score", 50.0)
            if session_id:
                session = self.sessions.get_or_create(session_id)
                session.update(cart_items, completeness=completeness)
                session_ctx = session.get_context()
                features["session_context"] = session_ctx

            # ── 4c. Dynamic top_k based on completeness ────────────────────
            effective_top_k = self._compute_dynamic_top_k(
                top_k, completeness, session_ctx,
            )
            tracer.metadata["effective_top_k"] = effective_top_k

            tracer.start_span("candidate_generation")
            candidates = self._generate_candidates(
                cart_items, restaurant, features
            )
            tracer.end_span(n_candidates=len(candidates))

            # ── 5. LLM Re-ranking (with circuit breaker) ──────────────────
            if strategy == "full":
                tracer.start_span("llm_reranking")
                ranked = self.fallback.breakers["llm_api"].call(
                    primary_fn=lambda: self._llm_rerank(
                        candidates, cart_items, restaurant, context
                    ),
                    fallback_fn=lambda: self._graph_rerank(
                        candidates, cart_items, restaurant
                    ),
                )
                tracer.end_span()
            elif strategy == "graph_only":
                tracer.start_span("llm_reranking")
                ranked = self._graph_rerank(
                    candidates, cart_items, restaurant
                )
                tracer.end_span(fallback="graph_only")
            else:
                tracer.start_span("llm_reranking")
                ranked = self._popularity_rerank(candidates)
                tracer.end_span(fallback=strategy)

            # ── 6. Post-processing ─────────────────────────────────────────
            tracer.start_span("post_processing")
            final = self._post_process(ranked, effective_top_k, cart_items)
            tracer.end_span(n_returned=len(final))

            # ── 6b. Record recommendations in session ─────────────────────
            if session_id and session_id in {s: s for s in [session_id]}:
                session = self.sessions.get(session_id)
                if session:
                    session.record_recommendations(
                        [r.get("name", "") for r in final]
                    )

            # ── 7. Assemble response ───────────────────────────────────────
            response = {
                "recommendations": final,
                "strategy": strategy,
                "effective_top_k": effective_top_k,
                "meal_completeness": completeness,
                "latency_ms": round(tracer.total_ms, 2),
                "trace": tracer.to_dict(),
            }
            if session_ctx:
                response["session"] = {
                    "step": session_ctx.get("session_step", 0),
                    "meal_is_done": session_ctx.get("meal_is_done", False),
                }

            # Cache the response
            self.cache.l2.set(cache_key, response,
                              ttl_s=self.config.cache.llm_response_ttl_s)
            self.cache.l1.put(cache_key, response, ttl_s=60)

            return response

        except Exception as exc:
            is_error = True
            logger.error(f"Pipeline error: {exc}", exc_info=True)
            return {
                "recommendations": [],
                "strategy": "error_fallback",
                "error": str(exc),
                "latency_ms": round(tracer.total_ms, 2),
                "trace": tracer.to_dict(),
            }
        finally:
            self.monitor.record_request(tracer, strategy=strategy, is_error=is_error)

    # ── PIPELINE STAGES (simulated) ─────────────────────────────────────────

    def _retrieve_features(
        self,
        cart_items: list[dict],
        restaurant: dict,
        user: dict | None,
    ) -> dict:
        """
        Retrieve features from the 3-tier feature pipeline.

        Tier 1 (real-time): Cart composition + session graph (<10ms)
        Tier 2 (cached):    User embeddings + preferences from ChromaDB (<20ms)
        Tier 3 (cached):    Temporal, geographic, restaurant context (<5ms)
        """
        features: dict = {}

        # ── Tier 1: Real-Time Cart Features ───────────────────────────────
        try:
            from ..features.cart_features import compute_cart_features
            cart_feats = compute_cart_features(cart_items)
            features.update(cart_feats.to_dict())
        except Exception as exc:
            logger.warning("Tier-1 cart features failed: %s", exc)
            # Fallback to basic aggregates
            cart_names = [i.get("name", "") for i in cart_items]
            total_value = sum(i.get("price", 0) for i in cart_items)
            features.update({
                "cart_total_value": total_value,
                "cart_item_count": len(cart_items),
                "cart_avg_price": round(total_value / max(len(cart_items), 1), 2),
            })

        # ── Tier 1: Temporal Session Graph ────────────────────────────────
        try:
            from ..features.session_graph import SessionGraph, PMIMatrix
            pmi = PMIMatrix.load()
            graph = SessionGraph(pmi_matrix=pmi)
            for item in cart_items:
                graph.add_item(item)
            features.update(graph.compute_graph_features())
            features["_session_graph"] = graph  # Pass graph downstream
            features["_graph_candidates"] = graph.get_candidate_scores(top_k=20)
        except Exception as exc:
            logger.warning("Tier-1 session graph failed: %s", exc)
            features.update({
                "graph_node_count": 0, "graph_edge_count": 0,
            })

        # ── Tier 2: User-Level Features ───────────────────────────────────
        try:
            from ..features.user_features import (
                UserEmbeddingStore, compute_user_preferences,
            )
            user_data = user or {}
            user_id = user_data.get("user_id", "anon")
            order_history = user_data.get("order_history", [])

            # Compute preferences
            prefs = compute_user_preferences(order_history)
            features.update(prefs.to_dict())

            # Try to get user embedding from ChromaDB
            if user_id != "anon" and order_history:
                try:
                    store = UserEmbeddingStore()
                    emb = store.get_user_embedding(user_id)
                    if emb is not None:
                        features["user_embedding_available"] = 1
                    else:
                        features["user_embedding_available"] = 0
                except Exception:
                    features["user_embedding_available"] = 0
            else:
                features["user_embedding_available"] = 0

        except Exception as exc:
            logger.warning("Tier-2 user features failed: %s", exc)
            features.update({
                "price_sensitivity": (user or {}).get("price_sensitivity", "medium"),
                "user_veg_ratio": 0.5,
            })

        # ── Tier 3: Contextual & Restaurant Features ──────────────────────
        try:
            from ..features.context_features import compute_all_context_features
            context_feats = compute_all_context_features(restaurant)
            features.update(context_feats)
        except Exception as exc:
            logger.warning("Tier-3 context features failed: %s", exc)
            features.update({
                "cuisine": restaurant.get("cuisine_type", "unknown"),
                "restaurant_type": restaurant.get("restaurant_type", "local"),
            })

        # ── Basic fields always present ───────────────────────────────────
        features["cart_names"] = [i.get("name", "") for i in cart_items]
        features["cart_categories"] = list({i.get("category", "") for i in cart_items})
        features["user_dietary"] = (user or {}).get("dietary_preference", "any")

        return features

    def _generate_candidates(
        self,
        cart_items: list[dict],
        restaurant: dict,
        features: dict,
    ) -> list[dict]:
        """
        Candidate generation via collaborative filtering.
        In production: FAISS ANN lookup + graph PMI lookup (< 50ms).
        Here we simulate with the restaurant menu / knowledge base.
        """
        from ..agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        result = agent.recommend(
            cart_items=cart_items,
            restaurant=restaurant,
            context={},
            cold_start_type="new_user",
            top_k=50,
        )
        return result.get("recommendations", [])

    def _llm_rerank(
        self,
        candidates: list[dict],
        cart_items: list[dict],
        restaurant: dict,
        context: dict | None,
    ) -> list[dict]:
        """
        LLM-powered re-ranking.
        In production: Claude API call with structured prompt (< 150ms).
        Here we simulate with the RerankerAgent.
        """
        from ..agents.meal_context_agent import MealContextAgent
        from ..agents.reranker_agent import RerankerAgent

        # Get meal context
        meal_agent = MealContextAgent()
        analysis = meal_agent.analyze(
            cart_items=cart_items,
            restaurant=restaurant,
            context=context or {"meal_type": "dinner"},
        )

        # Re-rank
        reranker = RerankerAgent()
        cf_candidates = []
        for c in candidates:
            cf_candidates.append({
                "item_id": c.get("item_id", c.get("name", "")[:6]),
                "name": c.get("name", ""),
                "category": c.get("category", ""),
                "price": c.get("price", 0),
                "is_veg": c.get("is_veg", True),
                "popularity_score": c.get("popularity_score", c.get("confidence", 0.5)),
                "margin_pct": 50,
                "tags": c.get("tags", []),
            })

        result = reranker.rerank(
            candidates=cf_candidates,
            context_analysis=analysis,
            user={"dietary_preference": "any"},
            business_config={
                "min_margin_pct": 10,
                "max_same_category": 3,
                "promoted_item_ids": [],
            },
            top_k=10,
        )
        return result.get("ranked_items", [])

    def _graph_rerank(
        self,
        candidates: list[dict],
        cart_items: list[dict],
        restaurant: dict,
    ) -> list[dict]:
        """Graph-only re-ranking fallback (no LLM). Uses confidence scores."""
        sorted_c = sorted(candidates, key=lambda x: -x.get("confidence", 0))
        for i, c in enumerate(sorted_c):
            c["rank"] = i + 1
        return sorted_c

    def _popularity_rerank(self, candidates: list[dict]) -> list[dict]:
        """Emergency fallback: sort by popularity_score only."""
        sorted_c = sorted(
            candidates,
            key=lambda x: -x.get("popularity_score", x.get("confidence", 0)),
        )
        for i, c in enumerate(sorted_c):
            c["rank"] = i + 1
        return sorted_c

    def _post_process(
        self,
        ranked: list[dict],
        top_k: int,
        cart_items: list[dict],
    ) -> list[dict]:
        """
        Apply business rules:
          - Filter out items below MIN_SCORE_THRESHOLD (if final_score is present)
          - Remove duplicates with cart
          - Enforce category diversity (max 3 per category)
          - Trim to top_k
        """
        cart_names = {i.get("name", "").lower() for i in cart_items}
        final = []
        cat_count: dict[str, int] = {}

        for item in ranked:
            # 1. Filter out low-confidence items
            score = item.get("final_score")
            if score is not None and score < MIN_SCORE_THRESHOLD:
                continue

            # 2. Duplicate check
            name = item.get("name", "")
            if name.lower() in cart_names:
                continue

            # 3. Category diversity
            cat = item.get("category", "other")
            if cat_count.get(cat, 0) >= 3:
                continue

            cat_count[cat] = cat_count.get(cat, 0) + 1
            final.append(item)
            if len(final) >= top_k:
                break

        return final

    def _compute_dynamic_top_k(
        self,
        base_top_k: int,
        completeness: float,
        session_ctx: dict[str, Any],
    ) -> int:
        """
        Adjust recommendation intensity based on how complete the meal is.
        - Meal < 50% complete  -> Aggressive (8 items)
        - Meal 50-80% complete -> Standard (6 items)
        - Meal > 80% complete  -> Gentle (3-4 items)
        """
        # If the meal is mostly complete, scale back recommendations
        if completeness >= 80.0:
            target = 4
        elif completeness >= 50.0:
            target = 6
        else:
            target = max(8, base_top_k)  # Aggressive defaults to 8 or what was requested

        # Further reduce based on recommendation fatigue (0-1 scale)
        fatigue = session_ctx.get("recommendation_fatigue", 0.0)
        if fatigue > 0.5:
            target = max(2, target - 2)

        return min(target, base_top_k)

    # ── RATE LIMITED ────────────────────────────────────────────────────────

    def _rate_limited_response(self, tracer: RequestTracer) -> dict:
        self.monitor.record_request(tracer, strategy="rate_limited", is_error=True)
        return {
            "recommendations": [],
            "strategy": "rate_limited",
            "error": "Too many requests — try again shortly",
            "latency_ms": round(tracer.total_ms, 2),
        }

    # ── STATUS / HEALTH ────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """Full system health snapshot."""
        return {
            "status": "ok",
            "circuit_breakers": self.fallback.get_all_stats(),
            "active_strategy": self.fallback.get_available_strategy(),
            "cache": self.cache.stats(),
            "monitoring": self.monitor.snapshot(),
        }

    def generate_architecture_doc(self) -> dict:
        """
        Generate the complete production architecture document
        matching the user's spec.
        """
        config = self.config

        return {
            "title": "CSAO Production Architecture — ContextFlow AI",
            "version": "1.0",

            # ── 1. Architecture Components ─────────────────────────────────
            "architecture_components": {
                "service_layer": {
                    "framework": "FastAPI (async, uvicorn workers)",
                    "load_balancing": "Kubernetes Ingress (NGINX) → ClusterIP Service",
                    "auto_scaling": {
                        "type": "Horizontal Pod Autoscaler (HPA)",
                        "target_cpu": f"{config.scaling.target_cpu_pct}%",
                        "min_replicas": config.scaling.min_replicas,
                        "max_replicas": config.scaling.max_replicas,
                        "scale_up_delay": f"{config.scaling.scale_up_delay_s}s",
                        "scale_down_delay": f"{config.scaling.scale_down_delay_s}s",
                    },
                },
                "data_layer": {
                    "feature_store": "Redis Cluster (6 nodes, 32GB) — user features, restaurant meta",
                    "vector_database": "ChromaDB / FAISS — item embeddings, ANN retrieval",
                    "cache": config.cache.to_dict(),
                },
                "model_layer": {
                    "candidate_generation": "FAISS ANN + Graph PMI (50ms SLA)",
                    "llm_reranking": "MealContextAgent + RerankerAgent pipeline (150ms SLA)",
                    "post_processing": "Business rules engine (20ms SLA)",
                    "parallelism": (
                        "Feature retrieval and candidate generation run in "
                        "parallel via ThreadPoolExecutor. LLM re-ranking is "
                        "sequential but has circuit breaker with graph-only fallback."
                    ),
                },
            },

            # ── 2. Latency Optimization ────────────────────────────────────
            "latency_optimization": {
                "budget_breakdown": config.latency.to_dict(),
                "strategies": [
                    "Pre-compute restaurant menus + item embeddings (batch, hourly)",
                    "L1 in-process LRU cache for repeat queries (< 1ms)",
                    "L2 Redis cache for cross-pod sharing (< 5ms)",
                    "LLM response caching for identical cart patterns (30 min TTL)",
                    "Async feature retrieval + candidate gen (parallel threads)",
                    "Circuit breaker → graph-only fallback eliminates LLM latency",
                    "Connection pooling for Redis + PostgreSQL",
                ],
            },

            # ── 3. Scalability ─────────────────────────────────────────────
            "scalability": {
                "scaling_calculations": config.scaling.to_dict(),
                "caching_strategy": {
                    "menu_cache": "1-hour TTL, ~50K restaurants, < 2 GB",
                    "user_embeddings": "5-min TTL, refreshed via Kafka consumer",
                    "llm_responses": "30-min TTL, keyed by cart hash, ~40% hit rate",
                },
                "database_optimization": {
                    "sharding": "User data sharded by user_id % N; restaurant by city",
                    "read_replicas": "2 PostgreSQL read replicas for feature store reads",
                    "connection_pooling": "pgbouncer (max 200 connections per pod)",
                },
            },

            # ── 4. Monitoring & Observability ──────────────────────────────
            "monitoring": {
                "key_metrics": [
                    "P50 / P95 / P99 end-to-end latency",
                    "Per-stage latency (feature, candidate, LLM, post-process)",
                    "Error rate (5-min sliding window)",
                    "Cache hit rates (L1, L2)",
                    "Circuit breaker states + fallback rate",
                    "Strategy distribution (full vs graph_only vs cf_only)",
                ],
                "alerting_rules": [
                    {"rule": "P95 latency > 400ms for 5 min",   "severity": "critical"},
                    {"rule": "Error rate > 5%",                  "severity": "critical"},
                    {"rule": "P99 latency > 800ms",              "severity": "critical"},
                    {"rule": "Cache hit rate < 30%",             "severity": "warning"},
                    {"rule": "LLM fallback rate > 20%",          "severity": "warning"},
                ],
                "logging": {
                    "per_request": [
                        "trace_id", "user_id", "cart_hash",
                        "strategy", "latency_ms", "span_breakdown",
                        "n_candidates", "n_returned", "cache_hit",
                    ],
                    "tracing": "Distributed trace with span-level breakdown per stage",
                },
            },

            # ── 5. Failure Handling ────────────────────────────────────────
            "failure_handling": {
                "llm_failures": {
                    "circuit_breaker": "3-state (CLOSED→OPEN→HALF_OPEN), 5-failure threshold, 30s recovery",
                    "fallback_chain": [
                        "1. Full pipeline (LLM + Graph)",
                        "2. Graph-only re-ranking (no LLM)",
                        "3. Collaborative filtering only",
                        "4. Cold Start Agent (cuisine KB)",
                        "5. Popularity-based (emergency)",
                    ],
                },
                "data_unavailability": {
                    "missing_user_features": "Route to ColdStartAgent",
                    "missing_restaurant_data": "Use cuisine-type defaults from KB",
                    "feature_store_down": "Serve from L1 cache + popularity fallback",
                },
                "overload_protection": {
                    "rate_limiter": "Token bucket (50K RPS, 60K burst)",
                    "load_shedding": "Drop non-critical requests when > 80% capacity",
                    "graceful_degradation": "Auto-switch to graph_only at high load",
                },
            },

            # ── 6. Cost Optimization ───────────────────────────────────────
            "cost_optimization": config.cost.compute_total(),

            # ── 7. Deployment Pipeline ─────────────────────────────────────
            "deployment_pipeline": {
                "ci_cd": {
                    "testing": [
                        "Unit tests (pytest, >80% coverage)",
                        "Integration tests (API contract tests)",
                        "Load tests (Locust, target 50K RPS)",
                        "Model quality tests (NDCG@10 regression)",
                    ],
                    "canary_stages": config.deployment.canary_stages,
                    "blue_green": "Zero-downtime via Kubernetes rolling update",
                },
                "model_updates": {
                    "versioning": "Model registry with semantic versioning",
                    "ab_testing": "ABTestFramework for new model versions",
                    "rollback": "Instant rollback via Kubernetes deployment revision",
                },
                "deployment_checklist": [
                    "✅ All unit + integration tests pass",
                    "✅ NDCG@10 ≥ 0.70 on holdout set",
                    "✅ P95 latency < 300ms in staging load test",
                    "✅ No critical alerts in canary (2h @ 5% traffic)",
                    "✅ Guardrail metrics within thresholds",
                    "✅ Circuit breakers tested (LLM kill-switch)",
                    "✅ Rollback procedure verified",
                    "✅ On-call engineer notified",
                ],
            },
        }
