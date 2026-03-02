#!/usr/bin/env python3
"""
batch_pipeline.py
=================
Batch feature refresh pipeline for CSAO.

Performs the following batch jobs (designed to run hourly / daily):
  1. PMI matrix update — recompute item co-occurrence statistics
  2. User embedding refresh — re-embed user histories into ChromaDB
  3. Restaurant feature refresh — aggregate performance metrics

Usage::

    python scripts/batch_pipeline.py --all
    python scripts/batch_pipeline.py --pmi-only
    python scripts/batch_pipeline.py --users-only
    python scripts/batch_pipeline.py --restaurants-only

In production, this would be orchestrated by Airflow DAGs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.session_graph import PMIMatrix

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PMI_OUTPUT = PROCESSED_DIR / "pmi_matrix.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  PMI MATRIX COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pmi_from_sessions(
    sessions: list[dict],
    min_count: int = 2,
    min_pmi: float = 0.05,
) -> PMIMatrix:
    """
    Compute Pointwise Mutual Information matrix from order sessions.

    PMI(a, b) = log2(P(a,b) / (P(a) * P(b)))

    Parameters
    ----------
    sessions : list[dict]
        Each session has 'cart_items' (list of item name strings) and
        optionally 'accepted_addons' (list of accepted add-on names).
    min_count : int
        Minimum co-occurrence count to include a pair.
    min_pmi : float
        Minimum PMI score to include.

    Returns
    -------
    PMIMatrix with computed scores.
    """
    logger.info("Computing PMI matrix from %d sessions...", len(sessions))

    # Count occurrences
    item_counts: dict[str, int] = defaultdict(int)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    total_sessions = len(sessions)

    for session in sessions:
        # Combine cart items and accepted add-ons into one "basket"
        cart = [_normalise(name) for name in session.get("cart_items", [])]
        addons = [_normalise(name) for name in session.get("accepted_addons", [])]
        basket = list(set(cart + addons))

        for item in basket:
            item_counts[item] += 1

        for i in range(len(basket)):
            for j in range(i + 1, len(basket)):
                a, b = min(basket[i], basket[j]), max(basket[i], basket[j])
                pair_counts[(a, b)] += 1

    # Compute PMI
    pmi_data: dict[str, dict[str, float]] = {}
    for (a, b), count in pair_counts.items():
        if count < min_count:
            continue

        p_ab = count / total_sessions
        p_a = item_counts[a] / total_sessions
        p_b = item_counts[b] / total_sessions

        if p_a == 0 or p_b == 0:
            continue

        pmi = math.log2(p_ab / (p_a * p_b))

        # Normalise to 0-1 range using NPMI: PMI / -log2(P(a,b))
        if p_ab > 0:
            npmi = pmi / (-math.log2(p_ab))
            npmi = max(0, min(1, (npmi + 1) / 2))  # shift to [0, 1]
        else:
            npmi = 0

        if npmi < min_pmi:
            continue

        if a not in pmi_data:
            pmi_data[a] = {}
        pmi_data[a][b] = round(npmi, 4)

    matrix = PMIMatrix(pmi_data)
    logger.info("PMI matrix computed: %d items, %d pairs",
                 len(pmi_data), sum(len(v) for v in pmi_data.values()))
    return matrix


def load_sessions_from_data() -> list[dict]:
    """
    Load session data for PMI computation.

    Tries to load from processed data, falls back to generating
    synthetic sessions for demonstration.
    """
    # Try loading processed data
    processed_file = PROCESSED_DIR / "sessions.json"
    if processed_file.exists():
        with open(processed_file, "r", encoding="utf-8") as f:
            sessions = json.load(f)
        logger.info("Loaded %d sessions from %s", len(sessions), processed_file)
        return sessions

    # Try loading the synthetic dataset
    parquet_file = PROCESSED_DIR / "cart_dataset.parquet"
    if parquet_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_file)
            sessions = _parquet_to_sessions(df)
            logger.info("Extracted %d sessions from parquet", len(sessions))
            return sessions
        except Exception as e:
            logger.warning("Failed to load parquet: %s", e)

    # Generate minimal synthetic sessions for demo
    logger.info("No existing data found — generating synthetic sessions for PMI")
    return _generate_demo_sessions()


def _parquet_to_sessions(df) -> list[dict]:
    """Convert parquet DataFrame to session list for PMI."""
    sessions = []
    if "session_id" in df.columns:
        for sid, group in df.groupby("session_id"):
            cart_items = []
            accepted = []
            for _, row in group.iterrows():
                name = _normalise(str(row.get("candidate_name", "")))
                if row.get("label", 0) == 1:
                    accepted.append(name)
                # Try to extract cart items
                cart_raw = row.get("cart_items", "")
                if isinstance(cart_raw, list):
                    cart_items.extend([_normalise(str(i)) for i in cart_raw])
                elif isinstance(cart_raw, str) and cart_raw:
                    cart_items.extend([_normalise(i.strip()) for i in cart_raw.split(",")])

            sessions.append({
                "cart_items": list(set(cart_items)),
                "accepted_addons": list(set(accepted)),
            })
    return sessions


def _generate_demo_sessions() -> list[dict]:
    """Generate synthetic sessions with common Indian food pairings."""
    import random
    rng = random.Random(42)

    combos = [
        (["biryani"], ["raita", "salan", "gulab_jamun"]),
        (["biryani"], ["raita", "lassi"]),
        (["butter_chicken"], ["naan", "jeera_rice", "dal_makhani"]),
        (["butter_chicken"], ["naan", "raita"]),
        (["paneer_tikka"], ["naan", "mint_chutney"]),
        (["dosa"], ["sambhar", "coconut_chutney", "vada"]),
        (["dosa"], ["sambhar", "filter_coffee"]),
        (["idli"], ["sambhar", "coconut_chutney"]),
        (["pizza"], ["garlic_bread", "coke"]),
        (["pizza"], ["fries", "coke"]),
        (["burger"], ["fries", "coke", "shake"]),
        (["noodles"], ["spring_roll", "manchurian"]),
        (["noodles"], ["sweet_corn_soup", "momos"]),
        (["fried_rice"], ["manchurian", "spring_roll"]),
        (["chole"], ["bhature", "lassi"]),
        (["pav_bhaji"], ["lassi", "masala_papad"]),
        (["thali"], ["papad", "gulab_jamun", "lassi"]),
        (["dal_makhani"], ["naan", "jeera_rice"]),
        (["biryani", "kebab"], ["raita", "salan"]),
        (["butter_chicken", "dal_makhani"], ["naan", "jeera_rice", "raita"]),
    ]

    sessions = []
    for _ in range(5000):
        cart, addons = rng.choice(combos)
        # Randomly include some add-ons
        accepted = [a for a in addons if rng.random() < 0.6]
        sessions.append({
            "cart_items": list(cart),
            "accepted_addons": accepted,
        })

    return sessions


def run_pmi_update() -> Path:
    """Run the PMI matrix batch update."""
    logger.info("═══ PMI Matrix Update ═══")
    start = time.time()

    sessions = load_sessions_from_data()
    matrix = compute_pmi_from_sessions(sessions)
    matrix.save(PMI_OUTPUT)

    elapsed = time.time() - start
    logger.info("PMI update complete in %.1fs → %s", elapsed, PMI_OUTPUT)
    return PMI_OUTPUT


# ═══════════════════════════════════════════════════════════════════════════════
#  USER EMBEDDING REFRESH
# ═══════════════════════════════════════════════════════════════════════════════

def run_user_embedding_refresh(max_users: int = 100) -> int:
    """
    Re-embed user histories and update ChromaDB.

    In production, this would process real user data from PostgreSQL.
    For demo, generates synthetic user profiles.
    """
    logger.info("═══ User Embedding Refresh ═══")
    start = time.time()

    from src.features.user_features import (
        UserEmbeddingStore, compute_user_preferences,
    )

    store = UserEmbeddingStore()
    import random
    rng = random.Random(42)

    # Generate synthetic users for demonstration
    cuisines = ["north_indian", "south_indian", "chinese", "continental",
                "fast_food", "mughlai", "street_food", "desserts"]
    meal_types = ["breakfast", "lunch", "dinner", "snack", "late_night"]
    cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
              "Kolkata", "Pune", "Ahmedabad", "Lucknow", "Jaipur"]

    count = 0
    now = time.time()

    for i in range(min(max_users, 100)):
        user_id = f"user_{i:04d}"

        # Generate synthetic order history
        n_orders = rng.randint(3, 30)
        orders = []
        for j in range(n_orders):
            ts = now - rng.randint(0, 90) * 86400  # random day in last 90d
            orders.append({
                "timestamp": ts,
                "total_value": rng.randint(150, 800),
                "cuisine": rng.choice(cuisines),
                "meal_type": rng.choice(meal_types),
                "is_veg": rng.random() < 0.5,
                "city": rng.choice(cities),
                "spice_level": rng.choice(["low", "medium", "high"]),
                "is_weekday": rng.random() < 0.7,
            })

        prefs = compute_user_preferences(orders, current_time=now)
        store.upsert_user(user_id, orders, prefs)
        count += 1

    elapsed = time.time() - start
    logger.info("User embedding refresh complete: %d users in %.1fs", count, elapsed)
    return count


# ═══════════════════════════════════════════════════════════════════════════════
#  RESTAURANT FEATURE REFRESH
# ═══════════════════════════════════════════════════════════════════════════════

def run_restaurant_refresh() -> int:
    """
    Refresh restaurant feature cache.

    In production, aggregates performance metrics from analytics DB.
    For demo, we pre-compute features for sample restaurants.
    """
    logger.info("═══ Restaurant Feature Refresh ═══")
    start = time.time()

    from src.features.context_features import compute_restaurant_features

    sample_restaurants = [
        {"name": "Biryani House", "restaurant_type": "local",
         "cuisine_type": "Mughlai", "avg_price": 280, "rating": 4.2},
        {"name": "Pizza Point", "restaurant_type": "chain",
         "cuisine_type": "Continental", "avg_price": 350, "rating": 4.0},
        {"name": "South Spice", "restaurant_type": "local",
         "cuisine_type": "South Indian", "avg_price": 180, "rating": 4.5},
        {"name": "Cloud Kitchen #1", "restaurant_type": "cloud_kitchen",
         "cuisine_type": "North Indian", "avg_price": 220, "rating": 3.8},
        {"name": "Premium Dining", "restaurant_type": "premium",
         "cuisine_type": "Continental", "avg_price": 600, "rating": 4.7},
    ]

    count = 0
    for rest in sample_restaurants:
        feats = compute_restaurant_features(rest)
        logger.debug("Restaurant '%s': %s", rest["name"], feats.to_dict())
        count += 1

    elapsed = time.time() - start
    logger.info("Restaurant refresh complete: %d restaurants in %.1fs", count, elapsed)
    return count


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser(
        description="CSAO Batch Feature Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Run all batch jobs")
    parser.add_argument("--pmi-only", action="store_true", help="Only update PMI matrix")
    parser.add_argument("--users-only", action="store_true", help="Only refresh user embeddings")
    parser.add_argument("--restaurants-only", action="store_true", help="Only refresh restaurant features")
    parser.add_argument("--max-users", type=int, default=100, help="Max users to process")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    run_all = args.all or not (args.pmi_only or args.users_only or args.restaurants_only)

    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║       CSAO Feature Engineering — Batch Pipeline          ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    start = time.time()

    if run_all or args.pmi_only:
        run_pmi_update()
        print()

    if run_all or args.users_only:
        run_user_embedding_refresh(max_users=args.max_users)
        print()

    if run_all or args.restaurants_only:
        run_restaurant_refresh()
        print()

    total = time.time() - start
    print(f"\n✅ Batch pipeline complete in {total:.1f}s")


if __name__ == "__main__":
    main()
