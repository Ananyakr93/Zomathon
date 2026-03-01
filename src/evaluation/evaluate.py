"""
evaluate.py
============
Comprehensive evaluation of the CSAO recommendation system.

Runs the full agent pipeline on test scenarios and computes:
  - NDCG@K, Precision@K, Recall@K, Hit@K
  - Acceptance rate simulation
  - Segment-level breakdown (user type, meal time, city, cold start)
  - Comparison vs popularity-based baseline
  - Latency profiling

Usage:
    python -m src.evaluation.evaluate
    python src/evaluation/evaluate.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.meal_context_agent import MealContextAgent
from src.agents.reranker_agent import RerankerAgent
from src.agents.cold_start_agent import ColdStartAgent

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  TEST SCENARIO GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def generate_test_scenarios(n: int = 200, seed: int = 42) -> list[dict]:
    """
    Generate diverse test scenarios covering all segments and edge cases.
    """
    rng = random.Random(seed)

    cuisines = [
        {"type": "North Indian", "items": [
            {"name": "Butter Chicken", "category": "main", "price": 320, "is_veg": False},
            {"name": "Paneer Tikka Masala", "category": "main", "price": 280, "is_veg": True},
            {"name": "Dal Makhani", "category": "main", "price": 220, "is_veg": True},
            {"name": "Chole Bhature", "category": "main", "price": 180, "is_veg": True},
        ]},
        {"type": "South Indian", "items": [
            {"name": "Masala Dosa", "category": "main", "price": 120, "is_veg": True},
            {"name": "Idli Sambhar", "category": "main", "price": 100, "is_veg": True},
            {"name": "Chicken Chettinad", "category": "main", "price": 350, "is_veg": False},
        ]},
        {"type": "Biryani Specialist", "items": [
            {"name": "Chicken Dum Biryani", "category": "main", "price": 349, "is_veg": False},
            {"name": "Veg Biryani", "category": "main", "price": 249, "is_veg": True},
            {"name": "Mutton Biryani", "category": "main", "price": 449, "is_veg": False},
        ]},
        {"type": "Chinese", "items": [
            {"name": "Hakka Noodles", "category": "main", "price": 220, "is_veg": True},
            {"name": "Chicken Manchurian", "category": "main", "price": 280, "is_veg": False},
            {"name": "Veg Fried Rice", "category": "main", "price": 200, "is_veg": True},
        ]},
        {"type": "Fast Food", "items": [
            {"name": "Classic Burger", "category": "main", "price": 180, "is_veg": False},
            {"name": "Margherita Pizza", "category": "main", "price": 299, "is_veg": True},
            {"name": "Chicken Wings", "category": "starter", "price": 250, "is_veg": False},
        ]},
    ]

    user_segments = ["new", "occasional", "frequent", "premium"]
    cities = [
        {"name": "Mumbai", "tier": "metro"},
        {"name": "Delhi", "tier": "metro"},
        {"name": "Bangalore", "tier": "metro"},
        {"name": "Lucknow", "tier": "tier_2"},
        {"name": "Jaipur", "tier": "tier_2"},
    ]
    meal_hours = {
        "breakfast": [7, 8, 9, 10],
        "lunch": [12, 13, 14],
        "dinner": [19, 20, 21],
        "late_night": [23, 0, 1],
    }

    scenarios = []
    for i in range(n):
        cuisine = rng.choice(cuisines)
        city = rng.choice(cities)
        segment = rng.choice(user_segments)
        meal_type = rng.choice(list(meal_hours.keys()))
        hour = rng.choice(meal_hours[meal_type])

        # Build cart (1-3 items)
        n_items = rng.randint(1, min(3, len(cuisine["items"])))
        cart = rng.sample(cuisine["items"], n_items)

        # Cold start scenarios (20% of test)
        is_cold_start = rng.random() < 0.20
        cold_start_type = rng.choice(["new_user", "new_restaurant", "none"])
        if not is_cold_start:
            cold_start_type = "none"

        scenarios.append({
            "scenario_id": f"eval_{i:04d}",
            "cart_items": cart,
            "restaurant": {
                "cuisine_type": cuisine["type"],
                "restaurant_type": rng.choice(["qsr", "chain", "independent", "premium"]),
            },
            "context": {
                "hour": hour,
                "meal_type": meal_type,
                "is_weekend": rng.random() < 0.3,
            },
            "user": {
                "user_id": f"U{i:05d}",
                "user_segment": segment,
                "dietary_preference": rng.choice(["vegetarian", "non_vegetarian", "any"]),
            },
            "city": city,
            "cold_start_type": cold_start_type,
            # Ground truth: items that "should" be recommended
            "expected_categories": _expected_categories(cuisine["type"], cart),
        })

    return scenarios


def _expected_categories(cuisine_type: str, cart: list[dict]) -> list[str]:
    """Determine what categories a good recommender should suggest."""
    cart_cats = {i.get("category", "") for i in cart}
    expected = []

    # Universal: always recommend a beverage if missing
    if "beverage" not in cart_cats and "drink" not in cart_cats:
        expected.append("beverage")

    # Cuisine-specific completeness
    if cuisine_type in ("North Indian",) and "bread" not in cart_cats:
        expected.append("bread")
    if cuisine_type in ("Biryani Specialist",) and "side" not in cart_cats:
        expected.append("side")  # raita
    if cuisine_type in ("Chinese",) and "soup" not in cart_cats:
        expected.append("soup")
    if "dessert" not in cart_cats:
        expected.append("dessert")

    return expected


# ═════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

def run_csao_pipeline(scenario: dict, top_k: int = 8) -> dict:
    """Run the full CSAO 3-agent pipeline on a scenario."""
    cart = scenario["cart_items"]
    restaurant = scenario["restaurant"]
    context = scenario["context"]

    # 1. Cold start check
    if scenario.get("cold_start_type", "none") != "none":
        agent = ColdStartAgent()
        result = agent.recommend(
            cart_items=cart,
            restaurant=restaurant,
            context=context,
            cold_start_type=scenario["cold_start_type"],
            top_k=top_k,
        )
        return {
            "items": result.get("recommendations", []),
            "strategy": f"cold_start_{scenario['cold_start_type']}",
        }

    # 2. Meal context analysis
    meal_agent = MealContextAgent()
    analysis = meal_agent.analyze(
        cart_items=cart,
        restaurant=restaurant,
        context=context,
    )

    # 3. Generate candidates via cold start KB (simulating CF)
    cold_agent = ColdStartAgent()
    candidates_result = cold_agent.recommend(
        cart_items=cart,
        restaurant=restaurant,
        context=context,
        cold_start_type="new_user",
        top_k=50,
    )
    candidates = candidates_result.get("recommendations", [])

    # 4. Re-rank
    reranker = RerankerAgent()
    cf_candidates = []
    for c in candidates:
        cf_candidates.append({
            "item_id": c.get("name", "")[:8],
            "name": c.get("name", ""),
            "category": c.get("category", ""),
            "price": c.get("price", 0),
            "is_veg": c.get("is_veg", True),
            "popularity_score": c.get("confidence", 0.5),
            "margin_pct": 50,
            "tags": [],
        })

    if cf_candidates:
        result = reranker.rerank(
            candidates=cf_candidates,
            context_analysis=analysis,
            user=scenario.get("user", {}),
            business_config={"min_margin_pct": 10, "max_same_category": 3, "promoted_item_ids": []},
            top_k=top_k,
        )
        return {"items": result.get("ranked_items", []), "strategy": "full"}

    return {"items": candidates[:top_k], "strategy": "fallback"}


def run_popularity_baseline(scenario: dict, top_k: int = 8) -> dict:
    """
    Popularity-based baseline: recommend top items by generic popularity.
    No meal context, no cultural intelligence, no personalization.
    """
    popular_items = [
        {"name": "Coke 500ml", "category": "beverage", "price": 60, "confidence": 0.8},
        {"name": "French Fries", "category": "side", "price": 120, "confidence": 0.75},
        {"name": "Gulab Jamun", "category": "dessert", "price": 99, "confidence": 0.7},
        {"name": "Garlic Bread", "category": "bread", "price": 129, "confidence": 0.65},
        {"name": "Mineral Water", "category": "beverage", "price": 30, "confidence": 0.6},
        {"name": "Brownie", "category": "dessert", "price": 149, "confidence": 0.55},
        {"name": "Cold Coffee", "category": "beverage", "price": 150, "confidence": 0.5},
        {"name": "Caesar Salad", "category": "side", "price": 180, "confidence": 0.45},
        {"name": "Masala Papad", "category": "starter", "price": 60, "confidence": 0.4},
        {"name": "Ice Cream", "category": "dessert", "price": 120, "confidence": 0.35},
    ]
    cart_names = {i.get("name", "").lower() for i in scenario["cart_items"]}
    filtered = [p for p in popular_items if p["name"].lower() not in cart_names]
    return {"items": filtered[:top_k], "strategy": "popularity_baseline"}


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    recommended: list[dict],
    expected_categories: list[str],
    k: int = 8,
) -> dict:
    """
    Compute recommendation quality metrics for a single scenario.
    """
    rec_cats = [r.get("category", "") for r in recommended[:k]]

    # Relevance: does recommended item fill an expected gap?
    relevant = [1 if cat in expected_categories else 0 for cat in rec_cats]
    n_relevant_total = len(expected_categories) if expected_categories else 1

    # Precision@K
    precision = sum(relevant) / max(len(relevant), 1)

    # Recall@K
    found = len(set(rec_cats) & set(expected_categories))
    recall = found / max(n_relevant_total, 1)

    # Hit@K
    hit = 1 if any(relevant) else 0

    # NDCG@K (with position discounting)
    dcg = sum(r / (i + 2) for i, r in enumerate(relevant))  # log2(i+2)
    ideal = sorted(relevant, reverse=True)
    idcg = sum(r / (i + 2) for i, r in enumerate(ideal))
    ndcg = dcg / max(idcg, 1e-9)

    # Simulated acceptance rate (items in expected categories + reasonable price)
    accepted = sum(1 for r in recommended[:k]
                   if r.get("category", "") in expected_categories
                   and r.get("price", 0) < 200)
    acceptance_rate = accepted / max(k, 1)

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "hit_at_k": hit,
        "ndcg_at_k": round(ndcg, 4),
        "acceptance_rate": round(acceptance_rate, 4),
        "n_recommendations": len(recommended[:k]),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SEGMENT BREAKDOWN
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_by_segment(
    results: list[dict],
    segment_key: str,
) -> dict[str, dict]:
    """Aggregate metrics by a segment dimension."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        seg_val = r.get(segment_key, "unknown")
        buckets[seg_val].append(r["metrics"])

    aggregated = {}
    for seg_val, metric_list in sorted(buckets.items()):
        aggregated[seg_val] = {
            "n_scenarios": len(metric_list),
            "precision_at_k": round(statistics.mean(m["precision_at_k"] for m in metric_list), 4),
            "recall_at_k": round(statistics.mean(m["recall_at_k"] for m in metric_list), 4),
            "hit_at_k": round(statistics.mean(m["hit_at_k"] for m in metric_list), 4),
            "ndcg_at_k": round(statistics.mean(m["ndcg_at_k"] for m in metric_list), 4),
            "acceptance_rate": round(statistics.mean(m["acceptance_rate"] for m in metric_list), 4),
        }
    return aggregated


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation(n_scenarios: int = 200, top_k: int = 8) -> dict:
    """
    Run the full evaluation pipeline:
      1. Generate test scenarios
      2. Run CSAO pipeline on each
      3. Run popularity baseline on each
      4. Compute metrics
      5. Aggregate by segment
      6. Compare CSAO vs baseline
    """
    print(f"Generating {n_scenarios} test scenarios...")
    scenarios = generate_test_scenarios(n_scenarios)

    csao_results = []
    baseline_results = []
    latencies: list[float] = []

    print(f"Running evaluation on {n_scenarios} scenarios...")
    for i, scenario in enumerate(scenarios):
        # CSAO pipeline (with latency measurement)
        t0 = time.perf_counter()
        csao_output = run_csao_pipeline(scenario, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        csao_metrics = compute_metrics(
            csao_output["items"],
            scenario["expected_categories"],
            k=top_k,
        )
        csao_results.append({
            "scenario_id": scenario["scenario_id"],
            "metrics": csao_metrics,
            "strategy": csao_output["strategy"],
            "user_segment": scenario["user"].get("user_segment", "unknown"),
            "meal_type": scenario["context"].get("meal_type", "unknown"),
            "cuisine": scenario["restaurant"].get("cuisine_type", "unknown"),
            "city_tier": scenario["city"].get("tier", "unknown"),
            "cold_start": scenario.get("cold_start_type", "none"),
        })

        # Baseline
        baseline_output = run_popularity_baseline(scenario, top_k=top_k)
        baseline_metrics = compute_metrics(
            baseline_output["items"],
            scenario["expected_categories"],
            k=top_k,
        )
        baseline_results.append({
            "scenario_id": scenario["scenario_id"],
            "metrics": baseline_metrics,
        })

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{n_scenarios}")

    # ── Aggregate overall ──────────────────────────────────────────────
    csao_overall = _aggregate_metrics([r["metrics"] for r in csao_results])
    baseline_overall = _aggregate_metrics([r["metrics"] for r in baseline_results])

    # ── Segment breakdowns ─────────────────────────────────────────────
    segments = {
        "by_user_segment": aggregate_by_segment(csao_results, "user_segment"),
        "by_meal_type": aggregate_by_segment(csao_results, "meal_type"),
        "by_cuisine": aggregate_by_segment(csao_results, "cuisine"),
        "by_city_tier": aggregate_by_segment(csao_results, "city_tier"),
        "by_cold_start": aggregate_by_segment(csao_results, "cold_start"),
    }

    # ── Latency stats ──────────────────────────────────────────────────
    latencies_sorted = sorted(latencies)
    latency_stats = {
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies_sorted[int(len(latencies_sorted) * 0.95)], 2),
        "p99_ms": round(latencies_sorted[int(len(latencies_sorted) * 0.99)], 2),
        "max_ms": round(max(latencies), 2),
    }

    # ── Lift calculation ───────────────────────────────────────────────
    lift = {}
    for key in csao_overall:
        base_val = baseline_overall.get(key, 0)
        csao_val = csao_overall.get(key, 0)
        if base_val > 0:
            lift[key] = round((csao_val - base_val) / base_val * 100, 1)
        else:
            lift[key] = 0.0

    return {
        "summary": {
            "n_scenarios": n_scenarios,
            "top_k": top_k,
        },
        "csao_overall": csao_overall,
        "baseline_overall": baseline_overall,
        "lift_vs_baseline_pct": lift,
        "segment_breakdown": segments,
        "latency": latency_stats,
    }


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    """Average metrics across all scenarios."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {
        k: round(statistics.mean(m[k] for m in metrics_list), 4)
        for k in keys
    }


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    results = run_evaluation(n_scenarios=200, top_k=8)

    # Print summary table
    print("\n" + "=" * 70)
    print("  CSAO EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'CSAO':>10} {'Baseline':>10} {'Lift':>10}")
    print("-" * 55)
    for key in ["precision_at_k", "recall_at_k", "hit_at_k", "ndcg_at_k", "acceptance_rate"]:
        csao_val = results["csao_overall"].get(key, 0)
        base_val = results["baseline_overall"].get(key, 0)
        lift_val = results["lift_vs_baseline_pct"].get(key, 0)
        print(f"  {key:<23} {csao_val:>9.4f} {base_val:>9.4f} {lift_val:>+9.1f}%")

    print(f"\n{'Latency':<25} {'Value':>10}")
    print("-" * 35)
    for key, val in results["latency"].items():
        print(f"  {key:<23} {val:>9.2f} ms")

    print(f"\n{'Segment':<25} {'NDCG@K':>10} {'Precision':>10} {'Acceptance':>10}")
    print("-" * 55)
    for seg_name, seg_data in results["segment_breakdown"]["by_user_segment"].items():
        print(f"  {seg_name:<23} {seg_data['ndcg_at_k']:>9.4f} "
              f"{seg_data['precision_at_k']:>9.4f} {seg_data['acceptance_rate']:>9.4f}")

    # Save full results
    out_path = Path("evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    main()
