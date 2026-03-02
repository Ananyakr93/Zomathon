"""
test_integration.py
===================
End-to-end integration test for the CSAO recommendation system.

Tests the full pipeline:
  1. Agent imports and initialization
  2. Inference pipeline load + recommend
  3. Sequential recommendations
  4. Health endpoint
  5. Cold start agent
  6. Reranker agent
  7. Meal context agent
  8. FastAPI app routes
"""

import sys
import time
import os
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = 0
FAIL = 0

def check(name: str, fn):
    global PASS, FAIL
    try:
        result = fn()
        if result:
            PASS += 1
            print(f"  [PASS] {name}")
        else:
            FAIL += 1
            print(f"  [FAIL] {name} -- returned False")
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {name} -- {type(e).__name__}: {e}")


def main():
    print("=" * 60)
    print("  CSAO END-TO-END INTEGRATION TEST")
    print("=" * 60)

    # ── 1. Module Imports ────────────────────────────────────────────
    print("\n── MODULE IMPORTS ──")

    check("Import MealContextAgent", lambda: (
        __import__("src.agents.meal_context_agent", fromlist=["MealContextAgent"]) and True
    ))
    check("Import RerankerAgent", lambda: (
        __import__("src.agents.reranker_agent", fromlist=["RerankerAgent"]) and True
    ))
    check("Import ColdStartAgent", lambda: (
        __import__("src.agents.cold_start_agent", fromlist=["ColdStartAgent"]) and True
    ))
    check("Import InferencePipeline", lambda: (
        __import__("src.serving.inference", fromlist=["InferencePipeline"]) and True
    ))
    check("Import CircuitBreaker", lambda: (
        __import__("src.serving.circuit_breaker", fromlist=["CircuitBreaker"]) and True
    ))
    check("Import CacheManager", lambda: (
        __import__("src.serving.cache_manager", fromlist=["CacheManager"]) and True
    ))
    check("Import ProductionConfig", lambda: (
        __import__("src.serving.production_config", fromlist=["ProductionConfig"]) and True
    ))
    check("Import Monitoring", lambda: (
        __import__("src.serving.monitoring", fromlist=["RequestMonitor"]) and True
    ))
    check("Import ABTestFramework", lambda: (
        __import__("src.experimentation.ab_test_framework", fromlist=["ABTestFramework"]) and True
    ))
    check("Import Metrics", lambda: (
        __import__("src.experimentation.metrics", fromlist=["compute_ndcg"]) and True
    ))
    check("Import Ranker", lambda: (
        __import__("src.models.ranker", fromlist=["train"]) and True
    ))
    check("Import Preprocessor", lambda: (
        __import__("src.data.preprocessor", fromlist=["engineer_features"]) and True
    ))

    # ── 2. Agent Construction ────────────────────────────────────────
    print("\n── AGENT CONSTRUCTION ──")

    from src.agents.meal_context_agent import MealContextAgent
    from src.agents.reranker_agent import RerankerAgent
    from src.agents.cold_start_agent import ColdStartAgent

    check("MealContextAgent instantiation", lambda: MealContextAgent() is not None)
    check("RerankerAgent instantiation", lambda: RerankerAgent() is not None)
    check("ColdStartAgent instantiation", lambda: ColdStartAgent() is not None)

    # ── 3. MealContextAgent ──────────────────────────────────────────
    print("\n── MEAL CONTEXT AGENT ──")

    mca = MealContextAgent()
    cart = [
        {"name": "Chicken Biryani", "category": "main", "price": 349, "is_veg": False},
    ]
    restaurant = {"cuisine_type": "Biryani Specialist", "price_tier": "mid"}
    context = {"hour": 13, "meal_type": "lunch"}

    def test_mca():
        result = mca.analyze(cart, restaurant, context)
        return (
            result.get("meal_type_identification") is not None and
            result.get("meal_completion_analysis") is not None
        )
    check("MealContextAgent.analyze() returns meal analysis", test_mca)

    # ── 4. ColdStartAgent ────────────────────────────────────────────
    print("\n── COLD START AGENT ──")

    csa = ColdStartAgent()

    def test_csa():
        result = csa.recommend(cart, restaurant, context=context, top_k=10)
        recs = result.get("recommendations", [])
        return len(recs) == 10

    check("ColdStartAgent.recommend() returns 10 items", test_csa)

    def test_csa_veg():
        veg_cart = [{"name": "Paneer Tikka", "category": "main", "price": 280, "is_veg": True}]
        result = csa.recommend(veg_cart, restaurant, context=context, top_k=10)
        recs = result.get("recommendations", [])
        return all(r.get("is_veg", False) for r in recs)

    check("ColdStartAgent respects veg diet (no non-veg recs)", test_csa_veg)

    def test_csa_diversity():
        result = csa.recommend(cart, restaurant, context=context, top_k=10)
        recs = result.get("recommendations", [])
        cats = {r.get("category") for r in recs}
        return len(cats) >= 2

    check("ColdStartAgent returns diverse categories (≥2)", test_csa_diversity)

    # ── 5. RerankerAgent ─────────────────────────────────────────────
    print("\n── RERANKER AGENT ──")

    ra = RerankerAgent()

    def test_reranker():
        candidates = [
            {"name": "Raita", "category": "side", "price": 50, "is_veg": True, "popularity_score": 0.9},
            {"name": "Gulab Jamun", "category": "dessert", "price": 80, "is_veg": True, "popularity_score": 0.8},
            {"name": "Coke 500ml", "category": "beverage", "price": 50, "is_veg": True, "popularity_score": 0.95},
        ]
        # RerankerAgent.rerank takes context_analysis (output of MealContextAgent)
        context_analysis = mca.analyze(cart, restaurant, context)
        result = ra.rerank(candidates, context_analysis)
        return isinstance(result, dict) and len(result.get("ranked_items", [])) > 0

    check("RerankerAgent.rerank() returns ranked results", test_reranker)

    # ── 6. Inference Pipeline ────────────────────────────────────────
    print("\n── INFERENCE PIPELINE ──")

    from src.serving.inference import InferencePipeline

    pipeline = InferencePipeline()

    def test_pipeline_load():
        pipeline.load()
        return pipeline._loaded

    check("InferencePipeline.load() succeeds", test_pipeline_load)

    def test_pipeline_recommend():
        recs = pipeline.recommend(
            [{"name": "Butter Chicken", "category": "main", "price": 320}],
            top_n=10,
        )
        return len(recs) == 10

    check("InferencePipeline.recommend() returns 10 items", test_pipeline_recommend)

    def test_pipeline_recommend_sequential():
        recs = pipeline.recommend_sequential(
            cart_history=[
                [{"name": "Butter Chicken", "category": "main", "price": 320}],
                [{"name": "Butter Chicken", "category": "main", "price": 320},
                 {"name": "Butter Naan", "category": "bread", "price": 50}],
            ],
            top_n=10,
        )
        return len(recs) >= 1

    check("InferencePipeline.recommend_sequential() works", test_pipeline_recommend_sequential)

    def test_pipeline_no_duplicates():
        recs = pipeline.recommend(
            [{"name": "Pizza", "category": "main", "price": 299}],
            top_n=10,
        )
        names = [r.addon_name for r in recs]
        return len(names) == len(set(names))

    check("No duplicate recommendations", test_pipeline_no_duplicates)

    def test_pipeline_excludes_cart():
        recs = pipeline.recommend(
            [{"name": "Chicken Biryani", "category": "main", "price": 349}],
            top_n=10,
        )
        rec_names = {r.addon_name.lower() for r in recs}
        return "chicken biryani" not in rec_names

    check("Cart items excluded from recommendations", test_pipeline_excludes_cart)

    def test_pipeline_health():
        info = pipeline.health()
        return info.get("status") in ("healthy", "ok") and info.get("model_loaded") is True

    check("InferencePipeline.health() returns healthy", test_pipeline_health)

    # ── 7. Latency Check ─────────────────────────────────────────────
    print("\n── LATENCY CHECK ──")

    def test_latency():
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            pipeline.recommend(
                [{"name": "Chicken Biryani", "category": "main", "price": 349}],
                top_n=10,
            )
            times.append((time.perf_counter() - t0) * 1000)
        p95 = sorted(times)[8]  # 95th percentile of 10 runs
        print(f"       P50: {sorted(times)[4]:.2f}ms, P95: {p95:.2f}ms")
        return p95 < 300  # must be under 300ms

    check("Latency P95 < 300ms (10 runs)", test_latency)

    # ── 8. Production Config ─────────────────────────────────────────
    print("\n── PRODUCTION CONFIG ──")

    from src.serving.production_config import ProductionConfig

    def test_prod_config():
        cfg = ProductionConfig()
        return (
            cfg.latency.total_budget_ms > 0 and
            cfg.scaling.peak_rps > 0 and
            cfg.availability_target > 0
        )

    check("ProductionConfig has valid values", test_prod_config)

    # ── 9. Circuit Breaker ───────────────────────────────────────────
    print("\n── CIRCUIT BREAKER ──")

    from src.serving.circuit_breaker import CircuitBreaker

    def test_circuit_breaker():
        cb = CircuitBreaker(name="test")
        # Should start CLOSED (enum value is 'closed')
        return cb._state.value == "closed"

    check("CircuitBreaker starts CLOSED", test_circuit_breaker)

    # ── SUMMARY ──────────────────────────────────────────────────────
    total = PASS + FAIL
    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS}/{total} PASSED  |  {FAIL} FAILURES")
    print("=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
